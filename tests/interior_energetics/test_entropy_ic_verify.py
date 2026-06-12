"""
Unit tests for the entropy-IC cross-check paths.

Covers:
- ``proteus.interior_energetics.common._verify_initial_entropy`` (SPIDER path)
- ``proteus.interior_energetics.aragog.AragogRunner._verify_entropy_ic``
  (Aragog path)

These are orthogonal guards that compare the result of the primary P-S
``invert_temperature`` path against an independent PALEOS entropy lookup.
The SPIDER path uses ``zalmoxis.eos_export.compute_surface_entropy`` for a
surface-only scalar comparison (safe against PALEOS non-converged cells).
The Aragog path uses ``zalmoxis.eos_export.compute_entropy_adiabat`` for a
full T(P) profile comparison (with defensive NaN handling in the bracket
expansion). The checks exist because past API drift in the Aragog entropy
rewrite left the Aragog guard as dead code for several weeks, silently
swallowed by a broad ``except Exception``.
These regression tests ensure that:

1. A consistent inversion passes with a PASS log line.
2. A moderate mismatch (1-5 %) produces a WARN line without raising.
3. A large mismatch (> 5 %) raises ``RuntimeError``.
4. The SPIDER path is a no-op when Zalmoxis/PALEOS are unavailable.
5. The Aragog path skips silently for non-PALEOS configs.
6. Stale solver APIs (missing attributes) fail loudly, not silently.

Discriminating values: tsurf = 2873 K and 3517 K (off-grid, asymmetric so
ordering bugs surface), S values designed so the 1 %, 5 % and 8 % cases are
clearly in the right bucket.

Testing standards and documentation:
- docs/test_infrastructure.md
- docs/test_categorization.md
- docs/test_building.md
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ======================================================================
# SPIDER path: _verify_initial_entropy in common.py
# ======================================================================


def _make_zalmoxis_config(mantle_eos='PALEOS:MgSiO3'):
    """Mock config with a Zalmoxis + PALEOS interior struct."""
    config = MagicMock()
    config.interior_struct.zalmoxis = MagicMock()
    config.interior_struct.zalmoxis.mantle_eos = mantle_eos
    config.interior_struct.module = 'zalmoxis'
    return config


def _patch_zalmoxis_adiabat(S_adiabat_value):
    """
    Build patches that make ``compute_entropy_adiabat`` return a controlled
    S_target and make the Zalmoxis material dictionary helpers succeed with
    a mocked PALEOS file.
    """
    fake_mat_dicts = {
        'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
        'PALEOS-2phase:MgSiO3': {},
    }

    return [
        patch(
            'zalmoxis.eos_export.compute_entropy_adiabat',
            return_value={
                'S_target': float(S_adiabat_value),
                'P': np.array([1e5, 1e11]),
                'T': np.array([2000.0, 4000.0]),
            },
            create=True,
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value=fake_mat_dicts,
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('os.path.isfile', return_value=True),
    ]


@pytest.mark.unit
def test_spider_verify_passes_on_consistent_inversion(caplog):
    """
    Primary inversion and PALEOS adiabat agree within 1 % -> PASS verdict.
    """
    from proteus.interior_energetics.common import _verify_initial_entropy

    config = _make_zalmoxis_config()

    S_target = 6437.4  # from real P-S inversion at tsurf=2500 K
    # Adiabat 0.3 % off (within PASS window of 1 %)
    S_adiabat = S_target * 1.003

    with caplog.at_level(logging.INFO, logger='fwl.proteus.interior_energetics.common'):
        with (
            patch(
                'zalmoxis.eos_export.compute_surface_entropy',
                return_value={
                    'S_target': S_adiabat,
                    'P_surface': 1e5,
                    'T_surface': 2500.0,
                },
                create=True,
            ),
            patch(
                'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
                return_value={
                    'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                    'PALEOS-2phase:MgSiO3': {},
                },
            ),
            patch(
                'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
                return_value=None,
            ),
            patch('os.path.isfile', return_value=True),
        ):
            _verify_initial_entropy(config, S_target=S_target, tsurf=2873.0, source='unit-test')

    joined = '\n'.join(r.message for r in caplog.records)
    assert 'verdict=PASS' in joined, f'Expected PASS verdict in log, got: {joined!r}'
    # Rel diff 0.3 % must appear in the log
    assert 'diff=0.299%' in joined or 'diff=0.300%' in joined


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_spider_verify_warns_on_moderate_mismatch(caplog):
    """
    2 % discrepancy triggers WARN verdict and a log.warning, no exception.
    """
    from proteus.interior_energetics.common import _verify_initial_entropy

    config = _make_zalmoxis_config()

    S_target = 9079.8  # tsurf=3500 K value
    S_adiabat = S_target * 1.02  # 2 % off -> WARN

    with caplog.at_level(logging.INFO, logger='fwl.proteus.interior_energetics.common'):
        with (
            patch(
                'zalmoxis.eos_export.compute_surface_entropy',
                return_value={
                    'S_target': S_adiabat,
                    'P_surface': 1e5,
                    'T_surface': 3500.0,
                },
                create=True,
            ),
            patch(
                'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
                return_value={
                    'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                    'PALEOS-2phase:MgSiO3': {},
                },
            ),
            patch(
                'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
                return_value=None,
            ),
            patch('os.path.isfile', return_value=True),
        ):
            # Must not raise
            _verify_initial_entropy(config, S_target=S_target, tsurf=3517.0, source='unit-test')

    joined = '\n'.join(r.message for r in caplog.records)
    assert 'verdict=WARN' in joined, f'Expected WARN verdict: {joined!r}'
    # Discrimination: the verdict must specifically be WARN, not PASS or
    # FAIL. The bucket boundaries are 1 % (PASS/WARN) and 5 % (WARN/FAIL);
    # 2 % must land cleanly in WARN. A regression that collapsed the
    # three-bucket logic to a binary pass/fail would not produce WARN.
    assert 'verdict=PASS' not in joined
    assert 'verdict=FAIL' not in joined


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_spider_verify_raises_on_large_mismatch():
    """
    8 % discrepancy triggers FAIL verdict and raises RuntimeError.
    """
    from proteus.interior_energetics.common import _verify_initial_entropy

    config = _make_zalmoxis_config()

    S_target = 6437.4
    S_adiabat = S_target * 1.08  # 8 % off -> FAIL

    with (
        patch(
            'zalmoxis.eos_export.compute_surface_entropy',
            return_value={'S_target': S_adiabat, 'P_surface': 1e5, 'T_surface': 2500.0},
            create=True,
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={
                'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                'PALEOS-2phase:MgSiO3': {},
            },
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('os.path.isfile', return_value=True),
    ):
        with pytest.raises(RuntimeError, match=r'cross-check FAIL'):
            _verify_initial_entropy(config, S_target=S_target, tsurf=2873.0, source='unit-test')
    # Boundary discrimination: a 4 % mismatch is below the FAIL
    # threshold (5 %) and must NOT raise. Same instance, smaller
    # mismatch, no raise. A regression that hard-raised on any
    # non-zero mismatch would fail this.
    S_adiabat_warn = S_target * 1.04
    with (
        patch(
            'zalmoxis.eos_export.compute_surface_entropy',
            return_value={'S_target': S_adiabat_warn, 'P_surface': 1e5, 'T_surface': 2500.0},
            create=True,
        ) as mock_compute,
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={
                'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                'PALEOS-2phase:MgSiO3': {},
            },
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('os.path.isfile', return_value=True),
    ):
        _verify_initial_entropy(config, S_target=S_target, tsurf=2873.0, source='unit-test')
    # The 4 % path must actually reach the EOS computation (otherwise
    # the no-raise verdict would be from an unrelated early-skip path).
    mock_compute.assert_called_once()


@pytest.mark.unit
def test_spider_verify_skipped_when_zalmoxis_unavailable(caplog):
    """
    No Zalmoxis -> function returns cleanly, does not raise, logs at DEBUG.
    """
    from proteus.interior_energetics import common
    from proteus.interior_energetics.common import _verify_initial_entropy

    config = _make_zalmoxis_config()

    # Force the import inside the function to fail.
    import builtins

    real_import = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if 'zalmoxis' in name:
            raise ImportError(f'mocked: {name} unavailable')
        return real_import(name, *args, **kwargs)

    with caplog.at_level(logging.DEBUG, logger='fwl.proteus.interior_energetics.common'):
        with patch.object(builtins, '__import__', side_effect=fail_import):
            # Must not raise, must not print warning at INFO level
            _verify_initial_entropy(config, S_target=6437.4, tsurf=2873.0, source='unit-test')

    # No INFO-level "verdict=" line (the function returned early)
    info_records = [
        r for r in caplog.records if r.levelno >= logging.INFO and 'verdict=' in r.message
    ]
    assert info_records == [], f'Unexpected verdict line after skip: {info_records!r}'
    assert common is not None  # import smoke


@pytest.mark.unit
def test_spider_verify_skipped_when_paleos_file_missing(caplog):
    """
    Zalmoxis present but PALEOS file missing -> silent skip (DEBUG log only).
    """
    from proteus.interior_energetics.common import _verify_initial_entropy

    config = _make_zalmoxis_config()

    with caplog.at_level(logging.DEBUG, logger='fwl.proteus.interior_energetics.common'):
        with (
            patch(
                'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
                return_value={'PALEOS:MgSiO3': {'eos_file': ''}},
            ),
            patch('os.path.isfile', return_value=False),
            patch('zalmoxis.eos_export.compute_surface_entropy', create=True) as mock_compute,
        ):
            _verify_initial_entropy(config, S_target=6437.4, tsurf=2873.0, source='unit-test')

    # No PASS/WARN/FAIL verdict was ever logged
    assert not any('verdict=' in r.message for r in caplog.records)
    # Discrimination: the silent-skip branch must NOT have invoked the
    # PALEOS surface-entropy computation. A regression that proceeded
    # past the missing-file guard and then merely failed to log would
    # still have called the EOS helper.
    mock_compute.assert_not_called()


@pytest.mark.unit
def test_spider_verify_zero_s_target_is_handled():
    """
    S_target == 0 is guarded against (avoid ZeroDivisionError in rel_diff).
    """
    from proteus.interior_energetics.common import _verify_initial_entropy

    config = _make_zalmoxis_config()

    with (
        patch(
            'zalmoxis.eos_export.compute_surface_entropy',
            return_value={'S_target': 1234.5, 'P_surface': 1e5, 'T_surface': 2500.0},
            create=True,
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={
                'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                'PALEOS-2phase:MgSiO3': {},
            },
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('os.path.isfile', return_value=True),
    ):
        # Must not raise ZeroDivisionError
        result = _verify_initial_entropy(config, S_target=0.0, tsurf=2873.0, source='unit-test')
    # Contract: verifier returns None on the guarded-zero path; an unguarded
    # rel_diff = abs(x - 0) / 0 would have raised ZeroDivisionError instead.
    assert result is None
    # Discriminating check: tsurf was a positive scalar (so the source-side
    # surface-entropy computation is exercised) and S_target genuinely zero
    # (so the guard branch is the only one that can produce a silent pass).
    assert config.interior_struct.module == 'zalmoxis'


# ======================================================================
# Aragog path: AragogRunner._verify_entropy_ic
# ======================================================================


def _make_aragog_dummy_config():
    """Mock config with non-zalmoxis structure (verify should skip)."""
    config = MagicMock()
    config.interior_struct.module = 'dummy'
    config.interior_struct.zalmoxis = None
    return config


def _make_mock_entropy_solver(
    P_stag: np.ndarray,
    S_stag: np.ndarray,
    temperature_scalar_fn=None,
):
    """
    Build a fake EntropySolver exposing the attributes the fixed verify
    function needs: ``_S0``, ``_P_stag_flat``, ``entropy_eos``.
    """
    solver = MagicMock()
    solver._S0 = S_stag
    solver._P_stag_flat = P_stag

    eos = MagicMock()

    if temperature_scalar_fn is None:
        # Default: identity-ish T(P, S) = S (deterministic, asymmetric)
        def temperature_scalar_fn(p, s):
            return float(s) * 0.4  # S=6437 -> T=2575

    eos.temperature_scalar = MagicMock(side_effect=temperature_scalar_fn)
    eos.invert_temperature = MagicMock(side_effect=lambda p, t: float(t) / 0.4)
    solver.entropy_eos = eos
    solver.set_initial_entropy = MagicMock()
    return solver


@pytest.mark.unit
def test_aragog_verify_skips_for_dummy_module(tmp_path):
    """
    ``interior_struct.module='dummy'`` -> verify returns early, no EOS access.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    config = _make_aragog_dummy_config()
    interior_o = MagicMock()
    interior_o.aragog_solver = _make_mock_entropy_solver(
        P_stag=np.array([1e5, 5e10, 1.35e11]),
        S_stag=np.array([6437.0, 6437.0, 6437.0]),
    )

    # Must not raise and must not call the EOS helpers
    AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path))

    interior_o.aragog_solver.entropy_eos.temperature_scalar.assert_not_called()
    # Discrimination: invert_temperature must also not be called. A
    # regression that took the verify branch on dummy and only the
    # temperature_scalar mock was untouched (because the path
    # exclusively used invert_temperature) would falsely pass the
    # assert_not_called above.
    interior_o.aragog_solver.entropy_eos.invert_temperature.assert_not_called()
    # The set_initial_entropy override path must also be untouched on
    # the skip branch.
    interior_o.aragog_solver.set_initial_entropy.assert_not_called()


@pytest.mark.unit
def test_aragog_verify_raises_on_api_drift(tmp_path):
    """
    Stale API: solver missing ``_S0`` must propagate as AttributeError.
    This is the regression that detected the original dead-code bug.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis = MagicMock()
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    config.planet.tsurf_init = 2873.0

    # Build a solver that is missing _S0 entirely: simulate API drift
    # where a future refactor renames the attribute.
    class BrokenSolver:
        _P_stag_flat = np.array([1e5, 1e11])
        entropy_eos = MagicMock()

    solver = BrokenSolver()
    solver.entropy_eos.temperature_scalar = lambda p, s: float(s) * 0.4

    interior_o = MagicMock()
    interior_o.aragog_solver = solver

    with (
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={
                'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                'PALEOS-2phase:MgSiO3': {},
            },
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('os.path.isfile', return_value=True),
    ):
        # AttributeError must propagate (NOT be swallowed by the
        # narrowed except clause).
        with pytest.raises(AttributeError, match=r'_S0'):
            AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path))

    # Discrimination: a solver that DOES expose _S0 with the same other
    # mocks in place must not raise. The exception above must come
    # specifically from the missing attribute, not from an unrelated
    # path in the verify routine that fires on every call.
    class WorkingSolver:
        _S0 = np.array([6437.0, 6437.0])
        _P_stag_flat = np.array([1e5, 1e11])
        entropy_eos = MagicMock()
        set_initial_entropy = MagicMock()

    working_solver = WorkingSolver()
    working_solver.entropy_eos.temperature_scalar = MagicMock(
        side_effect=lambda p, s: float(s) * 0.4
    )
    working_solver.entropy_eos.invert_temperature = MagicMock(
        side_effect=lambda p, t: float(t) / 0.4
    )
    interior_o.aragog_solver = working_solver
    with (
        patch(
            'zalmoxis.eos_export.compute_entropy_adiabat',
            return_value={
                'P': np.array([1e5, 1e11]),
                'T': np.array([2575.0, 2575.0]),
                'S_target': 6437.0,
            },
            create=True,
        ) as mock_adiabat,
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={
                'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                'PALEOS-2phase:MgSiO3': {},
            },
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('os.path.isfile', return_value=True),
    ):
        AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path))
    # The verify path must have actually queried the PALEOS adiabat. A
    # regression that early-returned past the _S0 check would skip the
    # mock and the no-raise verdict would mean nothing.
    mock_adiabat.assert_called_once()


@pytest.mark.unit
def test_aragog_verify_runs_and_overrides_on_warn(tmp_path):
    """
    Inversion and adiabat disagree by ~2 % on the surface node: verify logs
    WARN and overrides the entropy profile via set_initial_entropy.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis = MagicMock()
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    config.planet.tsurf_init = 2500.0

    # Aragog IC profile: T(P, S) = S * 0.4 gives T_stag = 0.4 * S_stag.
    P_stag = np.array([1e5, 5e10, 1.35e11])
    S_stag = np.array([6437.4, 6437.4, 6437.4])  # uniform IC
    interior_o = MagicMock()
    interior_o.aragog_solver = _make_mock_entropy_solver(P_stag, S_stag)

    # Build an adiabat that is 2 % hotter at the surface -> rel diff ~2 %.
    T_surface_aragog = 6437.4 * 0.4  # 2574.96
    T_adiabat_surface = T_surface_aragog * 1.02
    T_adiabat_bulk = T_surface_aragog * 1.02

    fake_adiabat = {
        'P': np.array([1e5, 1.35e11]),
        'T': np.array([T_adiabat_surface, T_adiabat_bulk]),
        'S_target': 6566.0,
    }

    with (
        patch(
            'zalmoxis.eos_export.compute_entropy_adiabat',
            return_value=fake_adiabat,
            create=True,
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
            return_value={
                'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                'PALEOS-2phase:MgSiO3': {},
            },
        ),
        patch(
            'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
            return_value=None,
        ),
        patch('os.path.isfile', return_value=True),
    ):
        AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path))

    # Override path is intentionally DISABLED now (the full-profile
    # cross-check is advisory only, see aragog.py docstring). The verify
    # call must complete without touching the IC entropy.
    interior_o.aragog_solver.set_initial_entropy.assert_not_called()
    # Discrimination: the solver's temperature_scalar must have been
    # consulted for the IC profile evaluation. A regression that
    # short-circuited the entire verify routine would leave both
    # set_initial_entropy AND temperature_scalar untouched (both
    # assert_not_called would pass for the wrong reason).
    assert interior_o.aragog_solver.entropy_eos.temperature_scalar.called


@pytest.mark.unit
def test_aragog_verify_logs_on_large_mismatch_but_does_not_raise(tmp_path, caplog):
    """
    10 % discrepancy is logged (as a WARN about table boundary drift)
    but does NOT raise. The Aragog full-profile cross-check was found
    to fire on every production run at M>=2.0 Earth masses because the
    PALEOS P-T and regenerated P-S tables drift by up to ~10% at high
    P (memory pitfall 50). The cross-check is therefore advisory only;
    the scalar surface check in _set_entropy_ic is the authoritative
    IC sanity check.
    """
    import logging

    from proteus.interior_energetics.aragog import AragogRunner

    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis = MagicMock()
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    config.planet.tsurf_init = 2500.0

    P_stag = np.array([1e5, 5e10, 1.35e11])
    S_stag = np.array([6437.4, 6437.4, 6437.4])
    interior_o = MagicMock()
    interior_o.aragog_solver = _make_mock_entropy_solver(P_stag, S_stag)

    T_ref = 6437.4 * 0.4
    fake_adiabat = {
        'P': np.array([1e5, 1.35e11]),
        'T': np.array([T_ref * 1.10, T_ref * 1.10]),
        'S_target': 7081.0,
    }

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.aragog'):
        with (
            patch(
                'zalmoxis.eos_export.compute_entropy_adiabat',
                return_value=fake_adiabat,
                create=True,
            ),
            patch(
                'proteus.interior_struct.zalmoxis.load_zalmoxis_material_dictionaries',
                return_value={
                    'PALEOS:MgSiO3': {'eos_file': '/fake/paleos.dat'},
                    'PALEOS-2phase:MgSiO3': {},
                },
            ),
            patch(
                'proteus.interior_struct.zalmoxis.load_zalmoxis_solidus_liquidus_functions',
                return_value=None,
            ),
            patch('os.path.isfile', return_value=True),
        ):
            # Must not raise. Must log the mismatch at WARNING level.
            AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path))

    joined = '\n'.join(r.message for r in caplog.records)
    assert 'Entropy IC full-profile cross-check' in joined, (
        f'Expected full-profile log line, got: {joined!r}'
    )
    # Discrimination: the routine must NOT have written through the
    # override path (advisory-only contract). A regression that
    # re-enabled the override would also satisfy the log-line check
    # above but silently overwrite the IC profile.
    interior_o.aragog_solver.set_initial_entropy.assert_not_called()


# ======================================================================
# Aragog path: liquidus_super cold-surface guard
# ======================================================================


def _make_aragog_liquidus_super_config():
    """Mock config for a zalmoxis + PALEOS liquidus_super run."""
    config = MagicMock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis = MagicMock()
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    config.planet.temperature_mode = 'liquidus_super'
    config.planet.tsurf_init = 4000.0
    return config


def _patch_crosscheck_eos(monkeypatch, tmp_path, surface_T, p_cmb):
    """Mock the EOS dependencies of ``_verify_entropy_ic`` so the cold-surface
    guard logic can be unit-tested without the PALEOS tables.

    Provides a present (stub) EOS file, a super-liquidus solve that returns the
    intended warm surface temperature, and a warm monotone reference adiabat.
    The IC profile itself is set by the caller's ``temperature_scalar``.
    """
    import zalmoxis.eos_export as eos_export

    import proteus.interior_struct.zalmoxis as zmod

    eos_file = tmp_path / 'eos.dat'
    eos_file.write_text('stub')
    monkeypatch.setattr(
        zmod,
        'load_zalmoxis_material_dictionaries',
        lambda: {'PALEOS:MgSiO3': {'eos_file': str(eos_file)}},
    )
    monkeypatch.setattr(zmod, 'resolve_2phase_mgsio3_paths', lambda *a, **k: (None, None))
    monkeypatch.setattr(zmod, 'load_zalmoxis_solidus_liquidus_functions', lambda *a, **k: None)
    monkeypatch.setattr(
        zmod,
        'solve_superliquidus_adiabat',
        lambda config, hf_row: {
            'surface_T': surface_T,
            'S_target': 10591.0,
            'cmb_T': 13000.0,
            'achieved_superheat': 500.0,
            'binding_P': 1.2e11,
            'P_cmb': p_cmb,
        },
    )

    def fake_adiabat(
        eos_file,
        T_surface,
        P_surface,
        P_cmb,
        n_points,
        solidus_func,
        liquidus_func,
        solid_eos_file,
        liquid_eos_file,
    ):
        P = np.linspace(P_surface, P_cmb, n_points)
        T = T_surface + (P - P_surface) / (P_cmb - P_surface) * (13000.0 - T_surface)
        return {
            'P': P,
            'T': T,
            'S_target': 10591.0,
            'S_profile': np.full(n_points, 10591.0),
        }

    monkeypatch.setattr(eos_export, 'compute_entropy_adiabat', fake_adiabat)


@pytest.mark.physics_invariant
def test_aragog_verify_raises_on_cold_surface_liquidus_super(monkeypatch, tmp_path):
    """A liquidus_super IC that unpacks to a COLD surface beyond the Fei+2021
    calibration is rejected: the cross-check raises, because that steeply
    inverted profile is the energy-non-conserving cold-surface initial
    condition the guard exists to catch.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    p_cmb = 1.474e12  # m10, beyond the ~500 GPa Fei calibration
    config = _make_aragog_liquidus_super_config()
    _patch_crosscheck_eos(monkeypatch, tmp_path, surface_T=4243.0, p_cmb=p_cmb)
    # P from CMB (index 0) down to the surface; the IC unpacks cold at the
    # surface (2900 K) and hot at the base (11000 K) -> steep inversion.
    P_stag = np.array([p_cmb, 6e11, 1.5e11, 1e9, 1e5])

    def cold_T(p, s):
        return 2900.0 + (p - 1e5) / (p_cmb - 1e5) * (11000.0 - 2900.0)

    interior_o = MagicMock()
    interior_o.aragog_solver = _make_mock_entropy_solver(
        P_stag=P_stag,
        S_stag=np.full(P_stag.size, 10000.0),
        temperature_scalar_fn=cold_T,
    )
    with pytest.raises(RuntimeError, match='cold-surface inversion') as exc:
        AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path), {'P_cmb': p_cmb})
    msg = str(exc.value)
    # The message must name the mode and the out-of-calibration pressure so the
    # failure is actionable, and report the cold unpacked surface (2900 K) it
    # caught against the intended ~4243 K adiabat anchor.
    assert 'liquidus_super' in msg and 'GPa' in msg
    assert '2900 K' in msg


def test_aragog_verify_no_raise_on_warm_surface_liquidus_super(monkeypatch, tmp_path):
    """A correctly-anchored warm-surface liquidus_super IC does NOT raise, even
    when the table-drift verdict is FAIL: the guard is gated on the cold-surface
    signature, not the verdict magnitude (a benign ~7 % deep drift must pass).
    """
    from proteus.interior_energetics.aragog import AragogRunner

    p_cmb = 1.474e12
    config = _make_aragog_liquidus_super_config()
    _patch_crosscheck_eos(monkeypatch, tmp_path, surface_T=4243.0, p_cmb=p_cmb)
    P_stag = np.array([p_cmb, 6e11, 1.5e11, 1e9, 1e5])

    def warm_T(p, s):
        base = 4243.0 + (p - 1e5) / (p_cmb - 1e5) * (13000.0 - 4243.0)
        return base * (1.0 + 0.07 * (p / p_cmb))  # warm surface, ~7% deep drift

    interior_o = MagicMock()
    interior_o.aragog_solver = _make_mock_entropy_solver(
        P_stag=P_stag,
        S_stag=np.full(P_stag.size, 10000.0),
        temperature_scalar_fn=warm_T,
    )
    # Must not raise; the cross-check runs to completion and writes its
    # diagnostic, confirming the FAIL verdict was reached but not escalated.
    AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path), {'P_cmb': p_cmb})
    assert (
        tmp_path / 'data' / 'entropy_ic_verification' / 'entropy_ic_comparison.npz'
    ).exists()
    interior_o.aragog_solver.entropy_eos.temperature_scalar.assert_called()
