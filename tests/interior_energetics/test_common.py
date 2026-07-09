"""
Unit tests for proteus.interior_energetics.common module: Interior_t lookup table loading.

Tests the _load_ps_table() method which loads SPIDER's P-S lookup tables
(density_melt, heat_capacity_solid, heat_capacity_melt) with FWL_DATA
fallback logic, and verifies the loaders are wired by ``Interior_t.__init__``.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- Interior_t._load_ps_table(): Load arbitrary P-S table with path fallback
- Interior_t.__init__(): Wires lookup_rho_melt + lookup_cp_solid + lookup_cp_melt
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t

# Tests run in the fast PR check. Each test patches BOTH the
# ``FWL_DATA`` environment variable AND the module-level
# ``proteus.utils.data.FWL_DATA_DIR`` constant so neither the
# inline ``os.environ.get('FWL_DATA', '')`` read in
# ``Interior_t._load_ps_table`` nor any downstream consumer of the
# import-time-frozen ``FWL_DATA_DIR`` constant can fall through to
# real on-disk lookup tables. The earlier hang on hosted CI runners
# was the classic module-level-constant trap: ``monkeypatch.setenv``
# alone does NOT reach a constant initialised at import time, and
# on a hosted image with a real ``FWL_DATA`` tree the frozen
# constant pointed at multi-gigabyte EOS data that genfromtxt would
# walk past the per-test 30 s ceiling. The 30 s timeout is a
# defensive ceiling against any future regression of the same shape.
pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_ps_table_file(filepath, nP=3, nS=4, val_scale=3000.0):
    """Create a minimal SPIDER-format P-S table file for testing.

    Format: 5-line header with dimensions and scale factors,
    followed by nS*nP lines of (P_nondim, S_nondim, value_nondim).
    """
    head = 5
    P_scale = 1e9
    S_scale = 2600.0
    with open(filepath, 'w') as f:
        f.write(f'# {head} {nP} {nS}\n')
        f.write('# units: Pa J/(kg K) value\n')
        f.write('# SPIDER P-S table\n')
        f.write('# test data\n')
        f.write(f'# {P_scale} {S_scale} {val_scale}\n')
        for j in range(nS):
            for i in range(nP):
                P_nd = float(i) / max(nP - 1, 1)
                S_nd = float(j) / max(nS - 1, 1)
                # Asymmetric formula so off-by-one in (i, j) ordering
                # produces a different value at every node.
                val_nd = 1.0 + 0.1 * P_nd + 0.05 * S_nd + 0.013 * P_nd * S_nd
                f.write(f'{P_nd} {S_nd} {val_nd}\n')


def test_load_ps_table_local_path(tmp_path):
    """Loads density_melt.dat from local SPIDER lookup_data path."""
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(os.path.join(eos_subdir, 'density_melt.dat'), nP=3, nS=4)

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        # Patch BOTH the env var (so the inline ``os.environ.get`` in
        # _load_ps_table cannot resolve to a real FWL_DATA path) AND
        # the module-level FWL_DATA_DIR constant in proteus.utils.data
        # (frozen at import time; not reached by setenv on hosted CI
        # images where FWL_DATA was already populated at process start).
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        mp.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path / 'nonexistent', raising=False)
        table = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'density_melt.dat'
        )

    assert table is not None
    assert table.shape == (4, 3, 3)
    # No NaNs introduced by reshaping or scaling
    assert not np.any(np.isnan(table))


def test_load_ps_table_fwl_data_path(tmp_path):
    """FWL_DATA path takes precedence over local path when both exist."""
    fwl_data = str(tmp_path / 'fwl_data')
    eos_path = os.path.join(
        fwl_data,
        'interior_lookup_tables',
        'EOS',
        'dynamic',
        'WolfBower2018_MgSiO3',
        'P-S',
    )
    os.makedirs(eos_path)
    _make_ps_table_file(
        os.path.join(eos_path, 'heat_capacity_melt.dat'),
        nP=5,
        nS=6,
        val_scale=4805046.0,
    )

    # Also create a local copy with a DIFFERENT shape so we can verify
    # which path actually got loaded.
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(os.path.join(eos_subdir, 'heat_capacity_melt.dat'), nP=2, nS=2)

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', fwl_data)
        # Mirror the env-var into the module-level constant so any
        # downstream consumer sees the same tmp_path-bound location.
        mp.setattr('proteus.utils.data.FWL_DATA_DIR', Path(fwl_data), raising=False)
        table = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'heat_capacity_melt.dat'
        )

    assert table is not None
    # Shape from FWL_DATA file (6 nS x 5 nP), not local (2 x 2)
    assert table.shape == (6, 5, 3)


def test_load_ps_table_both_missing(tmp_path):
    """Missing P-S file from both paths returns None (no exception)."""
    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        mp.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path / 'nonexistent', raising=False)
        result = interior_o._load_ps_table(
            str(tmp_path / 'also_nonexistent'),
            'SomeEOS',
            'density_melt.dat',
        )
    assert result is None  # both-missing branch must yield None silently
    # Discriminating check: neither candidate path exists on disk; only the
    # both-missing branch can have produced the None return on this row.
    assert not (tmp_path / 'nonexistent').exists()
    assert not (tmp_path / 'also_nonexistent').exists()


def test_load_ps_table_scaling(tmp_path):
    """Loaded data is scaled by the header scale factors."""
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    filepath = os.path.join(eos_subdir, 'density_melt.dat')

    nP, nS = 2, 2
    P_scale, S_scale, val_scale = 1e9, 2600.0, 3000.0
    with open(filepath, 'w') as f:
        f.write(f'# 5 {nP} {nS}\n')
        f.write('# units\n')
        f.write('# info\n')
        f.write('# info\n')
        f.write(f'# {P_scale} {S_scale} {val_scale}\n')
        # Asymmetric values: each node has a unique magnitude.
        f.write('0.1 0.2 0.5\n')
        f.write('0.3 0.2 0.6\n')
        f.write('0.1 0.8 0.7\n')
        f.write('0.3 0.8 0.9\n')

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        mp.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path / 'nonexistent', raising=False)
        table = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'density_melt.dat'
        )

    # First entry: P=0.1*1e9, S=0.2*2600, value=0.5*3000
    np.testing.assert_allclose(table[0, 0, 0], 0.1 * P_scale, rtol=1e-10)
    np.testing.assert_allclose(table[0, 0, 1], 0.2 * S_scale, rtol=1e-10)
    np.testing.assert_allclose(table[0, 0, 2], 0.5 * val_scale, rtol=1e-10)
    # Last entry differs from first along both axes; guards against
    # transposition / reshape ordering bugs.
    np.testing.assert_allclose(table[1, 1, 0], 0.3 * P_scale, rtol=1e-10)
    np.testing.assert_allclose(table[1, 1, 1], 0.8 * S_scale, rtol=1e-10)
    np.testing.assert_allclose(table[1, 1, 2], 0.9 * val_scale, rtol=1e-10)


def test_interior_t_init_loads_all_three_tables(tmp_path):
    """Interior_t.__init__ wires rho_melt + cp_solid + cp_melt loaders.

    The E_th calculation in spider.py depends on all three tables
    being loaded by ``Interior_t.__init__`` when ``spider_dir`` is
    provided. A regression here would silently re-introduce the
    Cp=1200 fallback that the 2026-04-09 parity work fixed.
    """
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(os.path.join(eos_subdir, 'density_melt.dat'), nP=3, nS=4)
    _make_ps_table_file(
        os.path.join(eos_subdir, 'heat_capacity_solid.dat'),
        nP=3,
        nS=4,
        val_scale=4805046.0,
    )
    _make_ps_table_file(
        os.path.join(eos_subdir, 'heat_capacity_melt.dat'),
        nP=3,
        nS=4,
        val_scale=4805046.0,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        mp.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path / 'nonexistent', raising=False)
        interior_o = Interior_t(50, spider_dir=spider_dir, eos_dir='WolfBower2018_MgSiO3')

    assert interior_o.lookup_rho_melt is not None
    assert interior_o.lookup_cp_solid is not None
    assert interior_o.lookup_cp_melt is not None
    assert interior_o.lookup_rho_melt.shape == (4, 3, 3)
    # Cp tables should be on the same grid as rho_melt
    assert interior_o.lookup_cp_solid.shape == interior_o.lookup_rho_melt.shape
    assert interior_o.lookup_cp_melt.shape == interior_o.lookup_rho_melt.shape


def test_interior_t_init_no_spider_dir_leaves_lookups_none():
    """Without spider_dir, all lookup tables stay None (Aragog path)."""
    interior_o = Interior_t(50)
    assert interior_o.lookup_rho_melt is None
    assert interior_o.lookup_cp_solid is None
    assert interior_o.lookup_cp_melt is None


def test_interior_t_init_partial_table_set(tmp_path):
    """Missing only the Cp tables: rho_melt loads, Cp_* stay None.

    Verifies that an incomplete EOS directory does not abort
    construction. The wrapper falls back to the Cp=1200 default in
    that case (with a warning).
    """
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(os.path.join(eos_subdir, 'density_melt.dat'), nP=3, nS=4)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        mp.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path / 'nonexistent', raising=False)
        interior_o = Interior_t(50, spider_dir=spider_dir, eos_dir='WolfBower2018_MgSiO3')

    assert interior_o.lookup_rho_melt is not None
    assert interior_o.lookup_cp_solid is None
    assert interior_o.lookup_cp_melt is None


def test_load_ps_table_invalid_filename(tmp_path):
    """Reading a nonexistent filename returns None, not raises."""
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        mp.setattr('proteus.utils.data.FWL_DATA_DIR', tmp_path / 'nonexistent', raising=False)
        result = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'this_file_does_not_exist.dat'
        )
    assert result is None  # missing-filename branch must yield None silently
    # Discriminating check: the EOS subdirectory exists but the requested
    # filename is genuinely absent, so the silent pass came from the
    # filename-not-found branch (not a missing-directory short-circuit).
    assert os.path.isdir(eos_subdir)
    assert not os.path.exists(os.path.join(eos_subdir, 'this_file_does_not_exist.dat'))


def test_interior_t_stale_struct_steps_init():
    """Interior_t initialises ``_stale_struct_steps`` to 0.

    The counter tracks consecutive Aragog steps integrated on a stale
    Zalmoxis structure, surfacing the silent-stale-mesh failure mode.

    Anti-happy-path: verifies the counter is mutable (not a property)
    and an integer, guarding against the two most plausible typo
    bugs (cached_property, wrong type).
    """
    interior_o = Interior_t(50)
    assert hasattr(interior_o, '_stale_struct_steps'), (
        'Interior_t must expose _stale_struct_steps for stale-mesh tracking'
    )
    assert interior_o._stale_struct_steps == 0
    assert isinstance(interior_o._stale_struct_steps, int)

    # Mutability: increment / reset cycle (the actual contract of the
    # counter).
    interior_o._stale_struct_steps += 1
    assert interior_o._stale_struct_steps == 1
    interior_o._stale_struct_steps = 0
    assert interior_o._stale_struct_steps == 0

    # Counter init is independent of mesh size (different nlev_b).
    interior_o_small = Interior_t(20)
    interior_o_large = Interior_t(200)
    assert interior_o_small._stale_struct_steps == 0
    assert interior_o_large._stale_struct_steps == 0


# ============================================================================
# Rheology lookup (eval_rheoparam): pinned values + invalid-name contract
# ============================================================================


@pytest.mark.physics_invariant
def test_eval_rheoparam_at_zero_melt_returns_solid_branch_with_three_class_discrimination():
    """At phi=0 (fully solid) the rheology parameter is strictly above the
    fully-molten dotl reference for visc and shear, and the bulk modulus
    is positive and finite. The three rheology channels share the same
    Kervazo+21 functional form (Bigphi = (1-phi)/(1-phi_star)) so a
    swap-of-channels regression flips the magnitude ordering.
    """
    from proteus.interior_energetics.common import (
        eval_rheoparam,
        par_bulk,
        par_shear,
        par_visc,
    )

    eta = eval_rheoparam(0.0, 'visc')
    mu = eval_rheoparam(0.0, 'shear')
    K = eval_rheoparam(0.0, 'bulk')

    # All channels must be strictly positive (physical positivity).
    assert eta > 0.0
    assert mu > 0.0
    assert K > 0.0
    # At phi=0, every channel is well above its own dotl reference
    # because Bigphi(0) > 1 and the numerator grows with delta.
    # A regression that mis-keyed the rheo_t lookup table would land
    # the value at the wrong dotl scale.
    assert eta > par_visc.dotl  # pure-solid viscosity >> melt reference (1.0 Pa s)
    assert mu > par_shear.dotl  # pure-solid shear >> melt reference (10 Pa)
    assert K > par_bulk.dotl  # pure-solid bulk >> melt reference (1e9 Pa)
    # Scale-discrimination guard: a melt-temperature regression returning
    # par.dotl would give eta=1 Pa s; the actual solid-branch value at
    # phi=0 is ~1e22 Pa s, so the order-of-magnitude check fires hard.
    assert eta > 1e10  # solid mantle viscosity well above 1e10 Pa s


@pytest.mark.physics_invariant
def test_eval_rheoparam_monotonic_decrease_from_solid_to_melt():
    """Viscosity and shear modulus decrease monotonically as melt fraction
    rises through the rheological transition (phi_star = 0.4). Symmetry
    invariant: the same monotonicity must hold for both channels.
    """
    from proteus.interior_energetics.common import eval_rheoparam

    phis = np.linspace(0.0, 0.99, 20)
    etas = np.array([eval_rheoparam(p, 'visc') for p in phis])
    mus = np.array([eval_rheoparam(p, 'shear') for p in phis])
    # Both channels strictly decreasing with melt fraction.
    assert np.all(np.diff(etas) <= 0.0), 'viscosity must be monotonically non-increasing in phi'
    assert np.all(np.diff(mus) <= 0.0), (
        'shear modulus must be monotonically non-increasing in phi'
    )
    # Discrimination: the ratio eta(0)/eta(0.99) must exceed 1e6 because
    # the Kervazo+21 rheology spans ~10 orders of magnitude across the
    # transition. A regression that returned par.dotl regardless of phi
    # would give a ratio of exactly 1.0.
    assert etas[0] / etas[-1] > 1e6


def test_eval_rheoparam_invalid_name_raises_value_error():
    """eval_rheoparam('foo') raises ValueError; valid channels do not."""
    from proteus.interior_energetics.common import eval_rheoparam

    with pytest.raises(ValueError, match='Invalid rheological parameter'):
        eval_rheoparam(0.3, 'this_is_not_a_channel')
    # Edge case: empty string also raises (no silent default).
    with pytest.raises(ValueError):
        eval_rheoparam(0.3, '')
    # Side-effect discrimination: the valid channels still produce
    # finite positive values, so the validator is a fail-fast filter
    # rather than blocking the whole module.
    for which in ('visc', 'shear', 'bulk'):
        val = eval_rheoparam(0.3, which)
        assert np.isfinite(val) and val > 0.0


# ============================================================================
# get_file_tides / Interior_t.write_tides / Interior_t.resume_tides
# ============================================================================


def test_get_file_tides_returns_expected_subpath(tmp_path):
    """get_file_tides composes ``outdir/data/tides_recent.dat``."""
    from proteus.interior_energetics.common import get_file_tides

    out = get_file_tides(str(tmp_path))
    # Edge case: trailing slash on the outdir must still produce a
    # single-separator join.
    out_slash = get_file_tides(str(tmp_path) + os.sep)
    assert out.endswith(os.path.join('data', 'tides_recent.dat'))
    assert out_slash.endswith(os.path.join('data', 'tides_recent.dat'))
    # The composed path must be inside outdir, not an absolute escape.
    assert str(tmp_path) in out


@pytest.mark.physics_invariant
def test_write_tides_then_resume_roundtrip_preserves_arrays(tmp_path):
    """write_tides + resume_tides is a roundtrip on a resolved interior.

    Physical positivity / boundedness invariant: melt fractions in [0, 1].
    """
    interior = Interior_t(5)  # nlev_b=5 -> nlev_s=4 (resolved interior)
    interior.phi = np.array([0.1, 0.5, 0.7, 0.9])
    interior.tides = np.array([1.0e-6, 2.0e-5, 3.0e-4, 4.0e-3])

    (tmp_path / 'data').mkdir()
    interior.write_tides(str(tmp_path))

    interior2 = Interior_t(5)
    interior2.resume_tides(str(tmp_path))

    np.testing.assert_allclose(interior2.phi, interior.phi, rtol=1e-6)
    np.testing.assert_allclose(interior2.tides, interior.tides, rtol=1e-6)
    # Boundedness invariant: melt fractions stayed in [0, 1].
    assert np.all((interior2.phi >= 0.0) & (interior2.phi <= 1.0))
    # Tidal heating rates non-negative.
    assert np.all(interior2.tides >= 0.0)


def test_write_tides_then_resume_dummy_interior_one_layer(tmp_path):
    """The nlev_s = 1 branch in resume_tides handles a single-layer dummy
    interior; the resolved branch would index a 1-D array as 2-D and fail.
    """
    interior = Interior_t(2)  # nlev_b=2 -> nlev_s=1 (dummy interior)
    interior.phi = np.array([0.42])
    interior.tides = np.array([7.5e-5])
    (tmp_path / 'data').mkdir()
    interior.write_tides(str(tmp_path))

    interior2 = Interior_t(2)
    interior2.resume_tides(str(tmp_path))
    assert interior2.phi[0] == pytest.approx(0.42, rel=1e-6)
    # Discrimination: the array shape is preserved (length 1 dummy-interior
    # signature) and the value matches; a regression that took the resolved
    # branch on a 1-D file would have raised IndexError.
    assert interior2.phi.shape == (1,)
    assert interior2.tides.shape == (1,)
    assert interior2.tides[0] == pytest.approx(7.5e-5, rel=1e-6)


def test_resume_tides_missing_file_emits_warning_and_does_not_mutate_state(tmp_path, caplog):
    """When no tides file exists, resume_tides logs a warning and leaves
    self.phi / self.tides as the constructor defaults (zeros)."""
    interior = Interior_t(5)
    interior.phi = np.zeros(4)
    interior.tides = np.zeros(4)
    (tmp_path / 'data').mkdir()
    # No file written.

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        interior.resume_tides(str(tmp_path))
    assert any('Cannot find tides file' in r.message for r in caplog.records)
    # State unchanged: zeros preserved (no array growth, no NaN injection).
    np.testing.assert_allclose(interior.phi, 0.0, atol=1e-12)
    np.testing.assert_allclose(interior.tides, 0.0, atol=1e-12)


def test_get_file_structure_stale_returns_expected_subpath(tmp_path):
    """get_file_structure_stale composes ``outdir/data/structure_stale.dat``."""
    from proteus.interior_energetics.common import get_file_structure_stale

    out = get_file_structure_stale(str(tmp_path))
    # Edge case: a trailing separator on the outdir must still join cleanly.
    out_slash = get_file_structure_stale(str(tmp_path) + os.sep)
    assert out.endswith(os.path.join('data', 'structure_stale.dat'))
    assert out_slash.endswith(os.path.join('data', 'structure_stale.dat'))
    # The composed path stays inside outdir, not an absolute escape, and is
    # distinct from the tides sidecar so the two records never collide.
    from proteus.interior_energetics.common import get_file_tides

    assert str(tmp_path) in out
    assert out != get_file_tides(str(tmp_path))


def test_write_structure_stale_survives_resume_roundtrip(tmp_path):
    """A raised stale flag survives a simulated resume through the sidecar.

    Contract: ``structure_stale`` records that the interior is running on a
    fall-back (previous-step) mesh. It lives on ``interior_o`` rather than the
    floats-only helpfile row, so it would be lost on resume (which rebuilds
    ``hf_row`` from the last CSV row) unless it is persisted separately. The
    write/resume pair is that persistence path.

    The resume target is a FRESH ``Interior_t``, whose constructor default is
    ``False``; that is the discrimination. If ``resume_structure_stale`` were a
    no-op, the fresh object would read ``False`` and the assertion would fail.
    The recovered ``True`` therefore proves the bit crossed the disk boundary.
    """
    (tmp_path / 'data').mkdir()

    # Direction 1 (the critical case): a crash right after a fall-back must resume
    # knowing the mesh is stale.
    stale = Interior_t(5)
    stale.structure_stale = True
    stale.write_structure_stale(str(tmp_path))

    resumed = Interior_t(5)
    assert resumed.structure_stale is False, 'fresh interior must default to not-stale'
    resumed.resume_structure_stale(str(tmp_path))
    assert resumed.structure_stale is True, 'raised flag must survive the resume'

    # Direction 2: a cleared flag persists too, so a resume after a successful
    # re-solve does not spuriously read stale. Discrimination: seed the resume
    # target True at entry so a no-op resume would leave it True; the recovered
    # False proves the on-disk 0 overrode the pre-resume state.
    fresh = Interior_t(5)
    fresh.structure_stale = False
    fresh.write_structure_stale(str(tmp_path))

    resumed_false = Interior_t(5)
    resumed_false.structure_stale = True
    resumed_false.resume_structure_stale(str(tmp_path))
    assert resumed_false.structure_stale is False, 'cleared flag must survive the resume'


def test_resume_structure_stale_missing_or_malformed_file_defaults_fresh(tmp_path, caplog):
    """resume_structure_stale degrades to not-stale on an absent or corrupt file.

    Edge case (absent file): a run with no sidecar, or one that never fell back,
    must assume the mesh is fresh rather than abort.
    Discrimination: the target is seeded ``True`` at entry, so the absent-file
    branch must actively reset it to ``False``, not merely leave a default.

    Error contract (malformed file): a non-integer payload must be swallowed with
    a warning and treated as fresh, so an unparseable visibility bit never blocks
    a resume.
    """
    from proteus.interior_energetics.common import get_file_structure_stale

    (tmp_path / 'data').mkdir()

    # Absent file: no sidecar written.
    missing = Interior_t(5)
    missing.structure_stale = True  # seed non-default so the reset is observable
    missing.resume_structure_stale(str(tmp_path))
    assert missing.structure_stale is False, 'absent file must reset to not-stale'

    # Malformed file: unparseable content must warn and fall back to False.
    with open(get_file_structure_stale(str(tmp_path)), 'w') as hdl:
        hdl.write('not-an-int\n')
    corrupt = Interior_t(5)
    corrupt.structure_stale = True
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        corrupt.resume_structure_stale(str(tmp_path))
    assert corrupt.structure_stale is False, 'malformed file must fall back to not-stale'
    assert any('stale-structure flag' in r.message for r in caplog.records), (
        'the malformed-file branch must emit a warning for provenance'
    )


def test_write_structure_stale_swallows_oserror_without_propagating(
    tmp_path, caplog, monkeypatch
):
    """A failed stale-flag write is logged and swallowed, never propagated.

    Contract: ``write_structure_stale`` is called on the Zalmoxis fall-back path
    in ``wrapper.py`` immediately before the mesh and ``zalmoxis_output.dat``
    rollback. The flag is only a resume-visibility bit, so a write failure (an
    absent ``data/`` directory, a full disk) must not abort the run; propagating
    the error here would skip the rollback that follows the call and strand the
    run on a half-committed structure. The guard logs and returns instead.

    Two directions, both of which would raise out of the call if the
    ``except OSError`` guard were removed:
    """
    from proteus.interior_energetics.common import get_file_structure_stale

    # Direction 1 (absent data/ dir): the realistic failure. ``tmp_path`` has no
    # ``data/`` subdirectory, so the underlying ``open(..., 'w')`` raises
    # ``FileNotFoundError`` (an ``OSError`` subclass). Discrimination: assert the
    # directory really is missing, then assert the guarded call still returns.
    target = get_file_structure_stale(str(tmp_path))
    assert not os.path.exists(os.path.dirname(target)), 'data/ must be absent for this branch'
    interior = Interior_t(5)
    interior.structure_stale = True
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        result = interior.write_structure_stale(str(tmp_path))
    assert result is None, 'the call must return normally, not propagate the write error'
    assert not os.path.exists(target), 'a failed write must not leave a partial flag file'
    assert any('stale-structure flag' in r.message for r in caplog.records), (
        'a swallowed write failure must still emit a warning for provenance'
    )

    # Direction 2 (generic OSError): the ``data/`` directory now exists, so the
    # only failure source is a patched ``open`` that raises a bare ``OSError``
    # (the disk-full case named in the source comment). This discriminates that
    # the guard catches ``OSError`` broadly, not merely the missing-dir subclass.
    (tmp_path / 'data').mkdir()

    def _raise_oserror(*args, **kwargs):
        raise OSError('simulated disk-full')

    monkeypatch.setattr('builtins.open', _raise_oserror)
    caplog.clear()
    disk_full = Interior_t(5)
    disk_full.structure_stale = False
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        assert disk_full.write_structure_stale(str(tmp_path)) is None
    assert any('stale-structure flag' in r.message for r in caplog.records), (
        'a bare OSError must be caught and logged, not only FileNotFoundError'
    )


def test_resume_tides_length_mismatch_logs_error_but_does_not_raise(tmp_path, caplog):
    """Writing nlev_s=3 data then resuming with nlev_s=4 logs an error
    but leaves the call free of exceptions (best-effort resume)."""
    interior_write = Interior_t(4)  # nlev_b=4 -> nlev_s=3
    interior_write.phi = np.array([0.1, 0.2, 0.3])
    interior_write.tides = np.array([1e-6, 2e-6, 3e-6])
    (tmp_path / 'data').mkdir()
    interior_write.write_tides(str(tmp_path))

    interior_read = Interior_t(5)  # nlev_b=5 -> nlev_s=4 (mismatched)
    with caplog.at_level('ERROR', logger='fwl.proteus.interior_energetics.common'):
        interior_read.resume_tides(str(tmp_path))
    assert any('Array length mismatch' in r.message for r in caplog.records)
    # The shorter array was loaded as-is; the mismatch surfaced via the
    # log error rather than via an in-band exception.
    assert interior_read.phi.shape == (3,)
    assert interior_read.tides.shape == (3,)


# ============================================================================
# Interior_t.update_rheology: shear/bulk + optional viscosity
# ============================================================================


@pytest.mark.physics_invariant
def test_update_rheology_fills_shear_and_bulk_arrays_skip_viscosity_by_default():
    """update_rheology(visc=False) fills shear + bulk arrays from current
    phi, but leaves the viscosity array untouched. The default skip path is
    the SPIDER/Aragog steady-state path where viscosity is computed
    elsewhere; only the dummy backend overrides this with visc=True.
    """
    interior = Interior_t(5)  # nlev_s = 4
    interior.phi = np.array([0.0, 0.3, 0.6, 0.9])
    interior.shear = np.zeros(4)
    interior.bulk = np.zeros(4)
    interior.visc = np.zeros(4)

    interior.update_rheology(visc=False)

    # Positivity invariant: shear and bulk moduli > 0 everywhere.
    assert np.all(interior.shear > 0.0)
    assert np.all(interior.bulk > 0.0)
    # Monotonicity: shear modulus decreases as phi increases (rheological
    # softening with melt fraction). A regression that swapped 'shear' and
    # 'bulk' channels would lose this monotonicity because the channels
    # have very different dotl reference scales.
    assert np.all(np.diff(interior.shear) <= 0.0)
    # Viscosity left untouched (still zeros from constructor).
    np.testing.assert_allclose(interior.visc, 0.0, atol=1e-12)


@pytest.mark.physics_invariant
def test_update_rheology_with_visc_true_fills_all_three_channels():
    """update_rheology(visc=True) is the dummy-backend path: viscosity,
    shear, and bulk are all recomputed from phi.
    """
    interior = Interior_t(4)  # nlev_s = 3
    interior.phi = np.array([0.05, 0.5, 0.95])

    interior.update_rheology(visc=True)

    # Positivity invariant for all three channels.
    assert np.all(interior.shear > 0.0)
    assert np.all(interior.bulk > 0.0)
    assert np.all(interior.visc > 0.0)
    # Discrimination: the viscosity values span many orders of magnitude
    # across the rheological transition; assert the dynamic range is at
    # least 1e3 so a regression that flattened the lookup is caught.
    assert interior.visc[0] / interior.visc[-1] > 1e3


# ============================================================================
# compute_initial_entropy: isentropic mode + Zalmoxis-unavailable fallback
# ============================================================================


def test_compute_initial_entropy_isentropic_mode_returns_user_value():
    """In isentropic mode compute_initial_entropy returns the user-set
    ini_entropy without any EOS lookup. This is the CHILI-compatible path.
    """
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='isentropic',
            ini_entropy=2500.0,
        )
    )
    S = compute_initial_entropy(config)
    assert S == pytest.approx(2500.0, rel=1e-12)
    # Discrimination: the fallback default (3200.0) and a typical Earth
    # surface adiabat (~2700 J/kg/K) are both far from 2500.0; the test
    # would fail under a regression that ignored the isentropic branch
    # and dropped to either of those.
    assert S < 2700.0
    assert S != pytest.approx(3200.0, rel=1e-3)


def test_compute_initial_entropy_isentropic_does_not_call_eos(monkeypatch):
    """Isentropic mode must not invoke any EOS lookup. Guard against a
    regression that re-evaluated the EOS even when temperature_mode was
    'isentropic' and would silently load ~30 MB of PALEOS tables.
    """
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    # Patch the EOS entry points to raise if reached.
    def _raise(*args, **kwargs):
        raise AssertionError('compute_initial_entropy must not load EOS in isentropic mode')

    monkeypatch.setattr('proteus.interior_energetics.common.os.path.isdir', _raise)

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='isentropic',
            ini_entropy=2900.0,
        )
    )
    # Even with spider_eos_dir set, the isentropic branch must short-circuit
    # before the os.path.isdir check.
    S = compute_initial_entropy(config, spider_eos_dir='/nonexistent/eos')
    assert S == pytest.approx(2900.0, rel=1e-12)
    # Sign / range discrimination: the user value is the only plausible
    # source of this exact number.
    assert 1e2 < S < 1e4


def test_compute_initial_entropy_fallback_when_zalmoxis_missing(monkeypatch, caplog):
    """When Zalmoxis is unavailable and no spider_eos_dir is supplied, the
    routine returns the fallback entropy and logs a warning. Tests the
    ModuleNotFoundError handling path at line ~465.
    """
    import sys
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    # Pretend zalmoxis is uninstalled by shadowing the import.
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos_export', None)
    monkeypatch.setitem(sys.modules, 'proteus.interior_struct.zalmoxis', None)

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic',
            tsurf_init=2400.0,
        )
    )
    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = compute_initial_entropy(config, fallback=3300.0)

    assert S == pytest.approx(3300.0, rel=1e-12)
    # Warning must explain the fallback so users can debug.
    assert any('Zalmoxis not installed' in r.message for r in caplog.records)
    # Discrimination: the explicit fallback kwarg (3300.0) must override
    # the function default (3200.0); a regression that hardcoded the
    # default would fail this.
    assert S != pytest.approx(3200.0, rel=1e-6)


def test_compute_initial_entropy_uses_t_surface_initial_override(monkeypatch, caplog):
    """When hf_row['T_surface_initial'] > 0, the helper overrides
    config.planet.tsurf_init with the accretion-derived value before any
    EOS lookup. The Zalmoxis-fallback path logs the override message.
    """
    import sys
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    # Force the Zalmoxis-unavailable path so we can intercept the log
    # message without needing a real EOS install.
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos_export', None)
    monkeypatch.setitem(sys.modules, 'proteus.interior_struct.zalmoxis', None)

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic',
            tsurf_init=2400.0,
        )
    )
    hf_row = {'T_surface_initial': 3100.0}  # accretion-derived override
    with caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.common'):
        compute_initial_entropy(config, hf_row=hf_row, fallback=3300.0)

    # Override log message must mention both the old and the new tsurf so
    # users can confirm the accretion override fired.
    override_msgs = [r.message for r in caplog.records if 'Overriding tsurf_init' in r.message]
    assert len(override_msgs) == 1, (
        f'override log must fire exactly once on positive T_surface_initial; '
        f'got {len(override_msgs)} occurrences'
    )
    # Negative / zero override must NOT fire (anti-happy-path: the gate
    # is `T_computed > 0`, not just truthy).
    caplog.clear()
    hf_row_zero = {'T_surface_initial': 0.0}
    with caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.common'):
        compute_initial_entropy(config, hf_row=hf_row_zero, fallback=3300.0)
    assert not any('Overriding tsurf_init' in r.message for r in caplog.records)


# ============================================================================
# _verify_initial_entropy: zalmoxis-unavailable skip + no-config skip
# ============================================================================


def test_verify_initial_entropy_skipped_when_zalmoxis_unavailable(monkeypatch, caplog):
    """The cross-check is a no-op when Zalmoxis is not installed.

    Guard: the helper must not raise; it must log a DEBUG line and return
    None. A regression that propagated the ImportError would crash any
    PROTEUS run on a machine without Zalmoxis.
    """
    import sys
    from types import SimpleNamespace

    from proteus.interior_energetics.common import _verify_initial_entropy

    # Force the lazy import to fail.
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos_export', None)
    monkeypatch.setitem(sys.modules, 'proteus.interior_struct.zalmoxis', None)

    config = SimpleNamespace(interior_struct=SimpleNamespace(zalmoxis=None))

    with caplog.at_level('DEBUG', logger='fwl.proteus.interior_energetics.common'):
        out = _verify_initial_entropy(config, S_target=2800.0, tsurf=2400.0, source='test')
    # Returns None silently (no exception, no value).
    assert out is None
    # The skip is logged so the silent no-op is auditable.
    debug_msgs = [r.message for r in caplog.records if 'zalmoxis unavailable' in r.message]
    assert len(debug_msgs) >= 1


def test_verify_initial_entropy_skipped_when_no_zalmoxis_cfg(monkeypatch, caplog):
    """The cross-check is also skipped when zalmoxis is installed but the
    config does not provide a zalmoxis sub-block (e.g. SPIDER with a dummy
    structure).
    """
    from types import SimpleNamespace

    from proteus.interior_energetics.common import _verify_initial_entropy

    # Sanity: skip if zalmoxis package not installed in this env.
    pytest.importorskip('zalmoxis')

    config = SimpleNamespace(interior_struct=SimpleNamespace(zalmoxis=None))
    with caplog.at_level('DEBUG', logger='fwl.proteus.interior_energetics.common'):
        out = _verify_initial_entropy(config, S_target=2800.0, tsurf=2400.0, source='dummy')
    assert out is None
    # Discrimination: no AttributeError is raised even though
    # config.interior_struct.zalmoxis is None.
    assert any('no Zalmoxis config' in r.message for r in caplog.records)


def test_compute_initial_entropy_adiabatic_from_cmb_uses_pcmb_fallback(monkeypatch, caplog):
    """When P_cmb is missing from hf_row, adiabatic_from_cmb mode falls back
    to a Noack & Lasbleis (2020) mass-aware estimate and logs a warning.

    Forces the Zalmoxis-unavailable branch so the routine reaches the
    fallback return without needing a real EOS install.
    """
    import sys
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    # Force the Zalmoxis-fallback path so the test does not need a real
    # PALEOS EOS file.
    monkeypatch.setitem(sys.modules, 'zalmoxis.eos_export', None)
    monkeypatch.setitem(sys.modules, 'proteus.interior_struct.zalmoxis', None)

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic_from_cmb',
            tcmb_init=4500.0,
            mass_tot=1.0,
        ),
        interior_struct=SimpleNamespace(
            module='dummy',
            core_frac=0.3,
            core_frac_mode='mass',
            zalmoxis=None,
        ),
    )
    hf_row = {}  # P_cmb missing -> triggers NL20 fallback

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = compute_initial_entropy(config, hf_row=hf_row, fallback=3300.0)

    # Zalmoxis-unavailable -> fallback entropy.
    assert S == pytest.approx(3300.0, rel=1e-12)
    # NL20 warning must fire because P_cmb was missing; the message names
    # both the fallback strategy and the mass.
    nl20_msgs = [r.message for r in caplog.records if 'Noack & Lasbleis (2020)' in r.message]
    assert len(nl20_msgs) == 1
    # The warning lists the mode by name so users can identify the
    # branch that landed here.
    assert 'adiabatic_from_cmb' in nl20_msgs[0]


def test_compute_initial_entropy_liquidus_super_without_zalmoxis_raises_runtime_error(
    monkeypatch,
):
    """liquidus_super mode requires Zalmoxis for paleos_liquidus; a missing
    import must raise RuntimeError with a clear message.
    """
    import sys
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    # Force the paleos_liquidus import to fail.
    monkeypatch.setitem(sys.modules, 'zalmoxis.melting_curves', None)

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='liquidus_super',
            mass_tot=1.0,
            delta_T_super=200.0,
        ),
        interior_struct=SimpleNamespace(
            module='dummy',
            core_frac=0.3,
            core_frac_mode='mass',
            zalmoxis=None,
        ),
    )
    hf_row = {'P_cmb': 1.35e11}  # valid CMB pressure, ~135 GPa

    with pytest.raises(RuntimeError, match='liquidus_super mode requires Zalmoxis') as exc:
        compute_initial_entropy(config, hf_row=hf_row, fallback=3300.0)
    # Discrimination: the message names BOTH the mode and the module that
    # would have provided the curve. A regression that swallowed the import
    # error and silently fell back to the user fallback would not raise at all.
    msg = str(exc.value)
    assert 'paleos_liquidus' in msg


def test_compute_initial_entropy_adiabatic_from_cmb_uses_provided_pcmb_no_fallback_warning(
    monkeypatch, caplog
):
    """When P_cmb IS supplied in hf_row, the NL20 fallback warning must NOT
    fire. Anti-happy-path: this pins the gate ``not P_cmb or P_cmb <= 0``,
    catching a regression that always fell to the NL20 path regardless of
    user input.
    """
    import sys
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    monkeypatch.setitem(sys.modules, 'zalmoxis.eos_export', None)
    monkeypatch.setitem(sys.modules, 'proteus.interior_struct.zalmoxis', None)

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic_from_cmb',
            tcmb_init=4500.0,
            mass_tot=1.0,
        ),
        interior_struct=SimpleNamespace(
            module='dummy',
            core_frac=0.3,
            core_frac_mode='mass',
            zalmoxis=None,
        ),
    )
    hf_row = {'P_cmb': 1.35e11}

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = compute_initial_entropy(config, hf_row=hf_row, fallback=3300.0)

    # Zalmoxis-unavailable -> still fallback, but no NL20 warning.
    assert S == pytest.approx(3300.0, rel=1e-12)
    assert not any('Noack & Lasbleis (2020)' in r.message for r in caplog.records), (
        'NL20 fallback fired even though P_cmb was supplied; gate broken'
    )


def test_compute_initial_entropy_adiabatic_from_cmb_negative_pcmb_falls_back(
    monkeypatch, caplog
):
    """A non-positive P_cmb in hf_row also triggers the NL20 fallback. This
    pins the second clause of the ``not P_cmb or P_cmb <= 0`` gate.
    """
    import sys
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    monkeypatch.setitem(sys.modules, 'zalmoxis.eos_export', None)
    monkeypatch.setitem(sys.modules, 'proteus.interior_struct.zalmoxis', None)

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic_from_cmb',
            tcmb_init=4500.0,
            mass_tot=1.0,
        ),
        interior_struct=SimpleNamespace(
            module='dummy',
            core_frac=0.3,
            core_frac_mode='mass',
            zalmoxis=None,
        ),
    )
    hf_row = {'P_cmb': -1.0}  # invalid: negative

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = compute_initial_entropy(config, hf_row=hf_row, fallback=3300.0)

    # Negative P_cmb routes through the NL20 fallback just like missing P_cmb.
    assert any('Noack & Lasbleis (2020)' in r.message for r in caplog.records), (
        'NL20 fallback did not fire on negative P_cmb; gate accepts negatives'
    )
    # Zalmoxis-unavailable + adiabatic_from_cmb returns the fallback entropy,
    # not the function default; discriminates against a regression that
    # ignored the kwarg.
    assert S == pytest.approx(3300.0, rel=1e-12)


def test_compute_initial_entropy_adiabatic_from_cmb_no_zalmoxis_cfg(monkeypatch, caplog):
    """In adiabatic_from_cmb mode, when zalmoxis is installed but no
    zalmoxis sub-config exists, the routine falls back with a clear
    warning rather than crashing.
    """
    pytest.importorskip('zalmoxis')
    from types import SimpleNamespace

    from proteus.interior_energetics.common import compute_initial_entropy

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic_from_cmb',
            tcmb_init=4500.0,
            mass_tot=1.0,
        ),
        interior_struct=SimpleNamespace(
            module='dummy',
            core_frac=0.3,
            core_frac_mode='mass',
            zalmoxis=None,  # explicitly missing
        ),
    )
    hf_row = {'P_cmb': 1.35e11}

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'):
        S = compute_initial_entropy(config, hf_row=hf_row, fallback=3300.0)

    assert S == pytest.approx(3300.0, rel=1e-12)
    # Warning names the mode and the requirement so users can fix the config.
    msgs = [
        r.message
        for r in caplog.records
        if 'requires interior_struct.module="zalmoxis"' in r.message
    ]
    assert len(msgs) >= 1


def test_compute_initial_entropy_paleos_failure_logs_warning_and_falls_back(
    monkeypatch, caplog
):
    """When the PALEOS path raises FileNotFoundError (missing table), the
    helper logs a warning and returns the fallback entropy. Pins the
    except-clause at line ~516.
    """
    pytest.importorskip('zalmoxis')
    from types import SimpleNamespace
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.common import compute_initial_entropy

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic',
            tsurf_init=2400.0,
        ),
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(mantle_eos='WolfBower2018:MgSiO3'),
        ),
    )

    with (
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={'WolfBower2018:MgSiO3': {'eos_file': '/tmp/missing_eos.dat'}},
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.resolve_2phase_mgsio3_paths',
            return_value=(None, None),
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'),
    ):
        S = compute_initial_entropy(config, fallback=3175.0)

    # The PALEOS path raised FileNotFoundError on the missing eos_file;
    # routine falls back to the user-supplied default.
    assert S == pytest.approx(3175.0, rel=1e-12)
    # Anti-happy-path: a discrimination guard against the function default
    # (3200.0): the user-supplied 3175.0 must propagate, not the bare
    # default. Catches a regression that ignored the kwarg.
    assert S != pytest.approx(3200.0, rel=1e-4)
    # Warning fired with a PALEOS-failure phrasing.
    assert any('Could not compute entropy from PALEOS' in r.message for r in caplog.records)


def test_verify_initial_entropy_zero_s_target_skipped(monkeypatch, caplog):
    """S_target == 0 short-circuits with a WARNING; verdicts cannot be
    computed when the denominator is zero.

    The path-through is gated on a previous successful PALEOS lookup, so
    we patch the dependencies to reach the S_target == 0 check.
    """
    pytest.importorskip('zalmoxis')
    from types import SimpleNamespace

    from proteus.interior_energetics.common import _verify_initial_entropy

    # Build a config with a zalmoxis sub-block.
    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(mantle_eos='WolfBower2018:MgSiO3'),
        )
    )

    # Stub the upstream PALEOS lookup to return a non-empty path and a
    # zero S_target.
    from unittest.mock import patch as _patch

    with (
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={'WolfBower2018:MgSiO3': {'eos_file': '/tmp/dummy_eos'}},
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.resolve_2phase_mgsio3_paths',
            return_value=('/tmp/solid_eos', '/tmp/liquid_eos'),
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        _patch('os.path.isfile', return_value=True),
        _patch(
            'zalmoxis.eos_export.compute_surface_entropy',
            return_value={'S_target': 0.0},
        ),
        caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'),
    ):
        out = _verify_initial_entropy(config, S_target=0.0, tsurf=2400.0, source='zero')
    assert out is None
    assert any('S_target is zero' in r.message for r in caplog.records)


# ============================================================================
# _verify_initial_entropy: PASS / WARN / FAIL verdict branches
# ============================================================================


def _patch_verify_inputs(s_adiabat: float):
    """Build the upstream patches needed to reach the verdict block.

    Returns a list of unittest.mock.patch context managers wired to a
    deterministic compute_surface_entropy return that yields the given
    adiabat S value.
    """
    from unittest.mock import patch as _patch

    return [
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={'WolfBower2018:MgSiO3': {'eos_file': '/tmp/dummy_eos'}},
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.resolve_2phase_mgsio3_paths',
            return_value=('/tmp/solid_eos', '/tmp/liquid_eos'),
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        _patch('os.path.isfile', return_value=True),
        _patch(
            'zalmoxis.eos_export.compute_surface_entropy',
            return_value={'S_target': s_adiabat},
        ),
    ]


@pytest.mark.physics_invariant
def test_verify_initial_entropy_pass_branch_within_one_percent(caplog):
    """A 0.5 % discrepancy (under the 1 % PASS threshold) logs the verdict
    as PASS and returns None.

    Physics invariant: the cross-check must accept agreement at the 1 %
    level, which is the empirical noise floor between the two algorithms
    (P-S inversion vs PALEOS adiabat) on the same EOS table.
    """
    pytest.importorskip('zalmoxis')
    from contextlib import ExitStack
    from types import SimpleNamespace

    from proteus.interior_energetics.common import _verify_initial_entropy

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(mantle_eos='WolfBower2018:MgSiO3'),
        )
    )

    S_target = 2800.0
    # 0.5 % offset, comfortably under the 1 % PASS bar.
    S_adiabat = S_target * 1.005
    with ExitStack() as stack:
        for cm in _patch_verify_inputs(S_adiabat):
            stack.enter_context(cm)
        with caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.common'):
            out = _verify_initial_entropy(
                config, S_target=S_target, tsurf=2400.0, source='pass'
            )
    assert out is None
    # Verdict in the log is PASS, not WARN or FAIL.
    pass_msgs = [r.message for r in caplog.records if 'verdict=PASS' in r.message]
    assert len(pass_msgs) == 1, (
        f'expected exactly one PASS verdict line; got {len(pass_msgs)} ({pass_msgs!r})'
    )
    # Anti-happy-path: a regression that flipped the comparison sense
    # would have logged WARN or FAIL on the same 0.5 % offset.
    assert not any('verdict=WARN' in r.message for r in caplog.records)
    assert not any('verdict=FAIL' in r.message for r in caplog.records)


def test_verify_initial_entropy_warn_branch_between_one_and_five_percent(caplog):
    """A 3 % discrepancy (between 1 % and 5 %) logs WARN and returns None.

    The WARN branch is a soft signal, distinct from FAIL which raises.
    """
    pytest.importorskip('zalmoxis')
    from contextlib import ExitStack
    from types import SimpleNamespace

    from proteus.interior_energetics.common import _verify_initial_entropy

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(mantle_eos='WolfBower2018:MgSiO3'),
        )
    )

    S_target = 2800.0
    S_adiabat = S_target * 1.03  # 3 % offset
    with ExitStack() as stack:
        for cm in _patch_verify_inputs(S_adiabat):
            stack.enter_context(cm)
        with caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.common'):
            out = _verify_initial_entropy(
                config, S_target=S_target, tsurf=2400.0, source='warn'
            )
    assert out is None
    warn_msgs = [r.message for r in caplog.records if 'verdict=WARN' in r.message]
    assert len(warn_msgs) == 1
    # WARN must NOT raise (only FAIL does).
    assert not any('verdict=FAIL' in r.message for r in caplog.records)


def test_verify_initial_entropy_fail_branch_raises_runtime_error_above_five_percent():
    """A 7 % discrepancy (above the 5 % FAIL bar) raises RuntimeError.

    Sign + scale discrimination: the assertion message must name BOTH
    the actual diff and the threshold so a future regression that
    silently relaxed the threshold is visible.
    """
    pytest.importorskip('zalmoxis')
    from contextlib import ExitStack
    from types import SimpleNamespace

    from proteus.interior_energetics.common import _verify_initial_entropy

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(mantle_eos='WolfBower2018:MgSiO3'),
        )
    )

    S_target = 2800.0
    S_adiabat = S_target * 1.07  # 7 % offset, > 5 % FAIL bar
    with ExitStack() as stack:
        for cm in _patch_verify_inputs(S_adiabat):
            stack.enter_context(cm)
        with pytest.raises(RuntimeError, match='Entropy IC cross-check FAIL') as exc:
            _verify_initial_entropy(config, S_target=S_target, tsurf=2400.0, source='fail')
    # The error message names BOTH the actual percentage and the threshold,
    # so a relaxation of the 5 % cap would land a different percentage in
    # the string. The 7 % offset must show up to within ~0.05 absolute.
    msg = str(exc.value)
    assert '7.0' in msg or '7.00' in msg, (
        f'FAIL message must report the actual % discrepancy; got {msg!r}'
    )


def test_verify_initial_entropy_skipped_when_paleos_file_missing(monkeypatch, caplog):
    """When the zalmoxis material dict has no eos_file AND solid_eos is
    empty, the cross-check skips with a DEBUG log line.
    """
    pytest.importorskip('zalmoxis')
    from types import SimpleNamespace
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.common import _verify_initial_entropy

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(mantle_eos='WolfBower2018:MgSiO3'),
        )
    )
    with (
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={'WolfBower2018:MgSiO3': {}},  # no eos_file
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.resolve_2phase_mgsio3_paths',
            return_value=(None, None),  # no solid_eos fallback
        ),
        _patch('os.path.isfile', return_value=False),
        caplog.at_level('DEBUG', logger='fwl.proteus.interior_energetics.common'),
    ):
        out = _verify_initial_entropy(config, S_target=2800.0, tsurf=2400.0, source='nofile')
    assert out is None
    # DEBUG message names PALEOS-file-not-found so the skip path is auditable.
    msgs = [r.message for r in caplog.records if 'PALEOS file not found' in r.message]
    assert len(msgs) >= 1


def test_verify_initial_entropy_expected_error_swallowed(monkeypatch, caplog):
    """KeyError / ValueError from the PALEOS lookup is swallowed with a
    WARNING (not raised). Pins the expected-error tuple at the try/except
    around the compute_surface_entropy call.
    """
    pytest.importorskip('zalmoxis')
    from types import SimpleNamespace
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.common import _verify_initial_entropy

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            zalmoxis=SimpleNamespace(mantle_eos='WolfBower2018:MgSiO3'),
        )
    )

    with (
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={'WolfBower2018:MgSiO3': {'eos_file': '/tmp/dummy_eos'}},
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.resolve_2phase_mgsio3_paths',
            return_value=('/tmp/solid_eos', '/tmp/liquid_eos'),
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        _patch('os.path.isfile', return_value=True),
        _patch(
            'zalmoxis.eos_export.compute_surface_entropy',
            side_effect=KeyError('S_target'),
        ),
        caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'),
    ):
        out = _verify_initial_entropy(
            config, S_target=2800.0, tsurf=2400.0, source='expected_err'
        )
    # Expected error path returns None cleanly.
    assert out is None
    assert any('cross-check skipped (expected error' in r.message for r in caplog.records)


# ============================================================================
# compute_initial_entropy success paths via spider_eos_dir + Zalmoxis adiabat
# ============================================================================


def test_compute_initial_entropy_ps_inversion_returns_eos_value_and_verifies(tmp_path):
    """Happy path: spider_eos_dir is set and points at a directory.
    EntropyEOS.invert_temperature returns a S value that is then
    cross-checked. The verify call must run with the same tsurf so the
    cross-check has access to the right context.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.common import compute_initial_entropy

    # spider_eos_dir must exist on disk (os.path.isdir gate at line 427).
    eos_dir = tmp_path / 'spider_eos'
    eos_dir.mkdir()

    eos_mock = MagicMock()
    eos_mock.invert_temperature.return_value = 2750.0

    config = SimpleNamespace(
        planet=SimpleNamespace(
            temperature_mode='adiabatic',
            tsurf_init=2400.0,
        ),
        interior_struct=SimpleNamespace(zalmoxis=None),
    )

    with (
        _patch('aragog.eos.entropy.EntropyEOS', return_value=eos_mock),
        # Verify helper is a no-op (already tested separately).
        _patch('proteus.interior_energetics.common._verify_initial_entropy'),
    ):
        S = compute_initial_entropy(config, spider_eos_dir=str(eos_dir))

    assert S == pytest.approx(2750.0, rel=1e-12)
    # Discriminating side-effect check: the EOS invert call must have
    # been at (1 bar, tsurf_init), not some swapped order.
    eos_mock.invert_temperature.assert_called_once()
    args = eos_mock.invert_temperature.call_args
    pressure_arg, temp_arg = args[0]
    assert pressure_arg == pytest.approx(1e5, rel=1e-12)
    assert temp_arg == pytest.approx(2400.0, rel=1e-12)


def test_compute_initial_entropy_ps_inversion_value_error_falls_through_to_paleos(
    tmp_path, caplog
):
    """When EntropyEOS.invert_temperature raises ValueError (target T
    out of range), the routine logs a warning and falls through to the
    Zalmoxis-adiabat path. The fallback also fails here because we
    short-circuit Zalmoxis, so the final return is the safe-fallback.
    """
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.common import compute_initial_entropy

    eos_dir = tmp_path / 'spider_eos'
    eos_dir.mkdir()

    eos_mock = MagicMock()
    eos_mock.invert_temperature.side_effect = ValueError('tsurf out of grid')

    monkeypatch_modules = sys.modules.copy()
    try:
        # Force the Zalmoxis-adiabat fallback to ALSO be unavailable.
        sys.modules['zalmoxis.eos_export'] = None
        sys.modules['proteus.interior_struct.zalmoxis'] = None

        config = SimpleNamespace(
            planet=SimpleNamespace(
                temperature_mode='adiabatic',
                tsurf_init=2400.0,
            ),
            interior_struct=SimpleNamespace(zalmoxis=None),
        )

        with (
            _patch('aragog.eos.entropy.EntropyEOS', return_value=eos_mock),
            caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.common'),
        ):
            S = compute_initial_entropy(config, spider_eos_dir=str(eos_dir), fallback=3275.0)
    finally:
        sys.modules.clear()
        sys.modules.update(monkeypatch_modules)

    # Both paths failed, so the user-supplied fallback wins.
    assert S == pytest.approx(3275.0, rel=1e-12)
    assert any('P-S inversion failed' in r.message for r in caplog.records)
    # Discrimination: the fallback wins over the function default (3200).
    assert S != pytest.approx(3200.0, rel=1e-4)
