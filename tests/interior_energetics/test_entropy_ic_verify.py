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
            _verify_initial_entropy(
                config, S_target=S_target, tsurf=2873.0, source='unit-test'
            )

    joined = '\n'.join(r.message for r in caplog.records)
    assert 'verdict=PASS' in joined, f'Expected PASS verdict in log, got: {joined!r}'
    # Rel diff 0.3 % must appear in the log
    assert 'diff=0.299%' in joined or 'diff=0.300%' in joined


@pytest.mark.unit
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


@pytest.mark.unit
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
        ):
            _verify_initial_entropy(config, S_target=6437.4, tsurf=2873.0, source='unit-test')

    # No PASS/WARN/FAIL verdict was ever logged
    assert not any('verdict=' in r.message for r in caplog.records)


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
        _verify_initial_entropy(config, S_target=0.0, tsurf=2873.0, source='unit-test')


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

    # Override path must have been taken
    interior_o.aragog_solver.set_initial_entropy.assert_called_once()
    override_arg = interior_o.aragog_solver.set_initial_entropy.call_args[0][0]
    assert override_arg.shape == S_stag.shape
    # Overridden S at surface must be ~ T_adiabat_surface / 0.4 (from the
    # identity-ish invert_temperature mock)
    np.testing.assert_allclose(
        override_arg[0],
        T_adiabat_surface / 0.4,
        rtol=1e-6,
    )


@pytest.mark.unit
def test_aragog_verify_raises_on_fail(tmp_path):
    """
    10 % discrepancy must raise RuntimeError (not just warn).
    """
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
        with pytest.raises(RuntimeError, match=r'cross-check FAIL'):
            AragogRunner._verify_entropy_ic(config, interior_o, str(tmp_path))
