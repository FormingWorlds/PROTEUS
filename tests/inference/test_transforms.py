"""Unit tests for inference parameter transforms (normalize/unnormalize with log scaling).

Tests the ``proteus.inference.transforms`` module, which wraps botorch's
normalize/unnormalize to apply log10 scaling for parameters that span orders
of magnitude (e.g. P_surf, *_vmr).

Invariants exercised:
- Round-trip identity: unnormalize(normalize(x)) == x for all valid physical inputs.
- Log-scaling correctness: the geometric mean of log-scaled bounds maps to 0.5 in
  normalised space; the arithmetic mean does not.
- Positivity guard: ValueError is raised for non-positive bounds or values on a
  log-scaled parameter.
- No-log pass-through: when no parameter is logarithmic the transform reduces to
  botorch's plain normalize/unnormalize.

References:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest
import torch

import proteus.inference.transforms as tr_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Tolerance for numerical comparison. Round-tripping through log10 -> botorch
# normalize -> botorch unnormalize -> exp10 introduces ~1e-15 float64 error.
_RTOL = 1e-12

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

# 'P_surf' is a known logarithmic variable (variable_is_logarithmic returns True).
# 'planet.mass_tot' is a linear variable.
_LOG_KEY = 'P_surf'
_LIN_KEY = 'planet.mass_tot'


def _bounds(*pairs: tuple[float, float]) -> torch.Tensor:
    """Return a (2, d) bounds tensor from pairs of (lo, hi)."""
    lo = [p[0] for p in pairs]
    hi = [p[1] for p in pairs]
    return torch.tensor([lo, hi], dtype=torch.double)


# --------------------------------------------------------------------------
# _log_mask
# --------------------------------------------------------------------------


def test_log_mask_identifies_log_scaled_key():
    """_log_mask returns True only for keys registered as logarithmic.

    Discrimination: 'P_surf' is log-scaled; 'planet.mass_tot' is linear.
    A regression that returned all-True or all-False would fail one branch.
    """
    mask = tr_mod._log_mask([_LOG_KEY, _LIN_KEY])
    assert mask[0].item() is True
    assert mask[1].item() is False
    # Edge case: empty key list returns empty tensor
    empty = tr_mod._log_mask([])
    assert empty.shape == (0,)


def test_log_mask_recognises_vmr_suffix():
    """Parameters ending in '_vmr' are logarithmic (volumetric mixing ratios
    span many orders of magnitude). Discrimination: '_vmr' must trigger the
    log flag; a plain 'vmr' without underscore prefix must not.
    """
    mask = tr_mod._log_mask(['H2O_vmr', 'CO2_bar', 'struct_mass'])
    assert mask[0].item() is True  # _vmr suffix
    assert mask[1].item() is True  # _bar suffix
    assert mask[2].item() is False  # no suffix match


# --------------------------------------------------------------------------
# _log10_bounds
# --------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_log10_bounds_transforms_logarithmic_column_correctly():
    """_log10_bounds applies log10 only to the log-scaled column.

    P_surf bounds [1e-3, 1e3] should become [-3, 3] after log10.
    The linear column (planet.mass_tot) must remain unchanged.

    Discrimination: a regression that applied log10 to the wrong column
    or to all columns would produce incorrect bounds for the linear parameter.
    """
    bounds = _bounds((1e-3, 1e3), (0.5, 3.0))
    keys = [_LOG_KEY, _LIN_KEY]
    log_bounds, log_mask = tr_mod._log10_bounds(bounds, keys)

    assert log_mask[0].item() is True
    assert log_mask[1].item() is False

    # Log column: log10([1e-3, 1e3]) = [-3, 3]
    assert log_bounds[0, 0].item() == pytest.approx(-3.0, rel=_RTOL)
    assert log_bounds[1, 0].item() == pytest.approx(3.0, rel=_RTOL)
    # Exponent-error guard: a wrong formula using log2 would give ~-10, 10
    assert abs(log_bounds[0, 0].item() - (-10.0)) > 1.0

    # Linear column: must be untouched
    assert log_bounds[0, 1].item() == pytest.approx(0.5, rel=_RTOL)
    assert log_bounds[1, 1].item() == pytest.approx(3.0, rel=_RTOL)


def test_log10_bounds_raises_for_non_positive_bound():
    """_log10_bounds raises ValueError when a log-scaled parameter has a
    non-positive bound, because log10(0) is -inf and log10(<0) is NaN.

    Discrimination: the error must fire for the logarithmic parameter, not
    a linear one; a linear parameter with bound 0.0 must not raise.
    """
    # Non-positive lower bound on a log-scaled parameter
    bad_bounds = _bounds((0.0, 1e3), (0.5, 3.0))
    with pytest.raises(ValueError, match=_LOG_KEY) as exc_info:
        tr_mod._log10_bounds(bad_bounds, [_LOG_KEY, _LIN_KEY])
    # Error must name the offending key, not a generic message
    assert _LOG_KEY in str(exc_info.value)

    # A non-positive bound on a *linear* parameter must not raise; the
    # returned bounds for the linear column must pass through unchanged.
    linear_zero = _bounds((1e-3, 1e3), (0.0, 3.0))
    result_bounds, _ = tr_mod._log10_bounds(linear_zero, [_LOG_KEY, _LIN_KEY])
    assert result_bounds[0, 1].item() == pytest.approx(0.0, abs=1e-12)


def test_log10_bounds_no_log_parameters_returns_original():
    """When no key is logarithmic, _log10_bounds returns the original bounds
    tensor and an all-False mask.
    """
    bounds = _bounds((0.5, 3.0), (0.1, 0.9))
    log_bounds, log_mask = tr_mod._log10_bounds(bounds, [_LIN_KEY, 'corefrac'])
    assert not log_mask.any()
    assert torch.allclose(log_bounds, bounds)


# --------------------------------------------------------------------------
# normalize_parameters / unnormalize_parameters round-trip (key invariant)
# --------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_round_trip_linear_only():
    """normalize then unnormalize recovers the original value for purely
    linear parameters. Discrimination: an off-by-one indexing bug would
    swap columns; with asymmetric bounds the swapped-column value is far
    from the correct one.

    Linear bounds [0.5, 3.0] and [0.1, 0.9]: mid-point 1.75 and 0.5.
    """
    keys = [_LIN_KEY, 'corefrac']
    bounds = _bounds((0.5, 3.0), (0.1, 0.9))
    raw = torch.tensor([[1.75, 0.5]], dtype=torch.double)

    normed = tr_mod.normalize_parameters(raw, bounds, keys)
    recovered = tr_mod.unnormalize_parameters(normed, bounds, keys)

    assert torch.allclose(recovered, raw, rtol=_RTOL)
    # Discrimination: linear midpoint maps to 0.5 in normalised space
    assert normed[0, 0].item() == pytest.approx(0.5, rel=_RTOL)
    assert normed[0, 1].item() == pytest.approx(0.5, rel=_RTOL)


@pytest.mark.physics_invariant
def test_round_trip_log_scaled_parameter():
    """normalize then unnormalize recovers the original value for a
    log-scaled parameter (P_surf). The geometric mean of [1e-3, 1e3]
    is 1.0, which must normalise to exactly 0.5.

    Discrimination: using arithmetic mean (500.0) would normalise to
    ~0.9997 (very close to 1.0, not 0.5); a regression that bypassed
    the log scaling would produce that value.
    """
    keys = [_LOG_KEY]
    bounds = _bounds((1e-3, 1e3))
    geometric_mean = torch.tensor([[1.0]], dtype=torch.double)

    normed = tr_mod.normalize_parameters(geometric_mean, bounds, keys)
    recovered = tr_mod.unnormalize_parameters(normed, bounds, keys)

    # Geometric mean normalises to exactly 0.5 in log space
    assert normed[0, 0].item() == pytest.approx(0.5, rel=_RTOL)
    # Round-trip recovers the original value
    assert recovered[0, 0].item() == pytest.approx(1.0, rel=_RTOL)

    # Discrimination: arithmetic mean (500.0) would NOT normalise to 0.5
    arith_mean = torch.tensor([[500.0]], dtype=torch.double)
    normed_arith = tr_mod.normalize_parameters(arith_mean, bounds, keys)
    assert abs(normed_arith[0, 0].item() - 0.5) > 0.1


@pytest.mark.physics_invariant
def test_round_trip_mixed_linear_and_log():
    """Round-trip with one log-scaled and one linear parameter recovers
    both original values. Tests that the per-column masking is correct
    and the two columns are independent.

    Bounds: P_surf in [1e-2, 1e2], mass in [0.5, 5.0].
    Values: P_surf=1.0 (geometric mean -> 0.5 normalised),
            mass=2.75 (arithmetic midpoint -> 0.5 normalised).
    """
    keys = [_LOG_KEY, _LIN_KEY]
    bounds = _bounds((1e-2, 1e2), (0.5, 5.0))
    raw = torch.tensor([[1.0, 2.75]], dtype=torch.double)

    normed = tr_mod.normalize_parameters(raw, bounds, keys)
    recovered = tr_mod.unnormalize_parameters(normed, bounds, keys)

    # Both dimensions must round-trip
    assert torch.allclose(recovered, raw, rtol=_RTOL)
    # Both map to 0.5 at their respective midpoints
    assert normed[0, 0].item() == pytest.approx(0.5, rel=_RTOL)
    assert normed[0, 1].item() == pytest.approx(0.5, rel=_RTOL)
    # Sign guard: values are positive (log-scaled parameter P_surf)
    assert recovered[0, 0].item() > 0


def test_normalize_parameters_raises_for_non_positive_log_value():
    """normalize_parameters raises ValueError when a log-scaled parameter
    has a non-positive input value. Zero and negative pressures are
    unphysical and should fail loudly.
    """
    keys = [_LOG_KEY]
    bounds = _bounds((1e-3, 1e3))
    with pytest.raises(ValueError, match=_LOG_KEY):
        tr_mod.normalize_parameters(torch.tensor([[0.0]], dtype=torch.double), bounds, keys)
    with pytest.raises(ValueError, match=_LOG_KEY):
        tr_mod.normalize_parameters(torch.tensor([[-1.0]], dtype=torch.double), bounds, keys)


# --------------------------------------------------------------------------
# _pool_timeout (gen_D_init)
# --------------------------------------------------------------------------


def test_pool_timeout_scales_with_task_load():
    """_pool_timeout (from gen_D_init) returns a positive finite value when
    the per-child timeout is set, and None when the timeout is disabled.

    Discrimination: with 4 tasks and 2 workers (2 tasks per worker), the
    pool timeout is roughly 2 * per_child + 300 s. A regression that
    divided by zero or swapped the ratio would produce a wildly different
    value.
    """
    import proteus.inference.gen_D_init as init_mod
    import proteus.inference.objective as obj_mod

    orig_env = __import__('os').environ.get(obj_mod._CHILD_TIMEOUT_ENV)
    try:
        # Enabled: 4 tasks, 2 workers -> per_worker = ceil(4/2) = 2
        obj_mod.set_child_timeout(3600.0)
        timeout = init_mod._pool_timeout(n_tasks=4, n_workers=2)
        assert timeout is not None
        assert timeout > 0
        # Expected: 2 * 3600 + 300 = 7500
        assert timeout == pytest.approx(7500.0, rel=1e-9)
        # Discrimination: wrong formula (n_tasks * per_child) would give 14400
        assert abs(timeout - 14400.0) > 100.0

        # Disabled: per_child returns None -> pool_timeout returns None
        obj_mod.set_child_timeout(0.0)
        timeout_none = init_mod._pool_timeout(n_tasks=4, n_workers=2)
        assert timeout_none is None
    finally:
        if orig_env is None:
            __import__('os').environ.pop(obj_mod._CHILD_TIMEOUT_ENV, None)
        else:
            __import__('os').environ[obj_mod._CHILD_TIMEOUT_ENV] = orig_env


# --------------------------------------------------------------------------
# child_timeout_s / set_child_timeout (objective module)
# --------------------------------------------------------------------------


def test_child_timeout_defaults_and_disables(monkeypatch):
    """child_timeout_s returns DEFAULT_CHILD_TIMEOUT_S when the env var is
    absent, returns None when set to 0.0 or below (disabled), returns the
    value when positive, and falls back to the default on invalid text.

    Discrimination: a regression that never returned None would let the pool
    hang on a wedged worker; one that never returned the default would make
    the timeout unpredictable on fresh environments.
    """
    import proteus.inference.objective as obj_mod

    env_key = obj_mod._CHILD_TIMEOUT_ENV
    monkeypatch.delenv(env_key, raising=False)
    assert obj_mod.child_timeout_s() == pytest.approx(obj_mod.DEFAULT_CHILD_TIMEOUT_S)

    # Positive value passes through
    monkeypatch.setenv(env_key, '1800.0')
    assert obj_mod.child_timeout_s() == pytest.approx(1800.0)

    # Zero disables the timeout
    monkeypatch.setenv(env_key, '0.0')
    assert obj_mod.child_timeout_s() is None

    # Negative disables the timeout
    monkeypatch.setenv(env_key, '-1.0')
    assert obj_mod.child_timeout_s() is None

    # Invalid text falls back to default
    monkeypatch.setenv(env_key, 'not-a-number')
    assert obj_mod.child_timeout_s() == pytest.approx(obj_mod.DEFAULT_CHILD_TIMEOUT_S)


def test_set_child_timeout_stores_in_env(monkeypatch):
    """set_child_timeout writes the value to the env var so pool workers
    (which may be spawned and cannot inherit module state) can read it.

    Discrimination: a regression that stored the value only in a module
    global would fail in spawned worker processes.
    """
    import os

    import proteus.inference.objective as obj_mod

    monkeypatch.delenv(obj_mod._CHILD_TIMEOUT_ENV, raising=False)
    obj_mod.set_child_timeout(7200.0)
    assert os.environ.get(obj_mod._CHILD_TIMEOUT_ENV) == '7200.0'

    # None selects DEFAULT_CHILD_TIMEOUT_S
    obj_mod.set_child_timeout(None)
    assert os.environ.get(obj_mod._CHILD_TIMEOUT_ENV) == str(obj_mod.DEFAULT_CHILD_TIMEOUT_S)


# --------------------------------------------------------------------------
# run_proteus timeout branch
# --------------------------------------------------------------------------


def test_run_proteus_wraps_timeout_as_runtime_error(monkeypatch, tmp_path):
    """subprocess.TimeoutExpired is wrapped as RuntimeError with a 'timed out'
    message so the inference harness receives a consistent error type.

    Discrimination: a regression that re-raised the raw TimeoutExpired would
    break the except-RuntimeError handler in the BO worker loop.
    """
    import subprocess

    import proteus.inference.objective as obj_mod

    out_abs = tmp_path / 'sim'
    out_abs.mkdir()
    monkeypatch.setattr(obj_mod, 'get_proteus_directories', lambda _: {'output': str(out_abs)})
    monkeypatch.setattr(obj_mod, 'update_toml', lambda *_a, **_kw: None)

    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=['proteus'], timeout=1.0)

    monkeypatch.setattr(obj_mod.subprocess, 'run', _fake_run)

    with pytest.raises(RuntimeError, match='timed out') as exc_info:
        obj_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='ref.toml',
            output='dummy_output',
        )
    assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)


# --------------------------------------------------------------------------
# J failure-code gate
# --------------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_J_returns_bad_value_on_failure_status(monkeypatch):
    """When PROTEUS returns a status code in the 20-29 generic-error range,
    J returns BAD_OBJ_VALUE instead of computing the objective, so BO
    does not fit a GP against useless data.

    Discrimination: a correct run (status 10) must produce a real
    objective near zero or positive; a bad-status run must return
    BAD_OBJ_VALUE = -20.0, which is far below any legitimate objective.
    """
    import proteus.inference.objective as obj_mod

    good_obs = {'R_obs': 1.0}

    def fake_run_good(*args, **kwargs):
        return ({'R_obs': 1.0}, 10)  # status 10: success

    def fake_run_bad(*args, **kwargs):
        return ({'R_obs': 1.0}, 20)  # status 20: generic error

    x = torch.tensor([[0.5]], dtype=torch.double)
    parameters = [_LIN_KEY]

    # Good run: objective is computed (close to perfect match = high value)
    monkeypatch.setattr(obj_mod, 'run_proteus', fake_run_good)
    y_good = obj_mod.J(
        x,
        parameters=parameters,
        true_observables=good_obs,
        worker=0,
        iter=0,
        ref_config='ref.toml',
        output='out',
        failure_codes=[],
    )
    # At exact match, obj = -log10(EPS_CLIP) ~ 10
    assert y_good.item() > -1.0
    assert torch.isfinite(y_good).all()

    # Bad run: must return BAD_OBJ_VALUE
    monkeypatch.setattr(obj_mod, 'run_proteus', fake_run_bad)
    y_bad = obj_mod.J(
        x,
        parameters=parameters,
        true_observables=good_obs,
        worker=0,
        iter=0,
        ref_config='ref.toml',
        output='out',
        failure_codes=[],
    )
    assert y_bad.item() == pytest.approx(obj_mod.BAD_OBJ_VALUE, rel=1e-9)
    # Discrimination: BAD_OBJ_VALUE is far below any real objective
    assert y_bad.item() < y_good.item() - 10.0


@pytest.mark.physics_invariant
def test_J_returns_bad_value_for_user_failure_code(monkeypatch):
    """User-supplied failure_codes cause J to return BAD_OBJ_VALUE for
    the matching status code, even if it is not in the default bad range.

    Discrimination: status 96 without failure_codes=[96] must allow a
    real objective to be computed; status 96 with failure_codes=[96] must
    return BAD_OBJ_VALUE.
    """
    import proteus.inference.objective as obj_mod

    # Status code of 96 is not assigned
    def fake_run(*args, **kwargs):
        return ({'R_obs': 1.0}, 96)

    monkeypatch.setattr(obj_mod, 'run_proteus', fake_run)
    x = torch.tensor([[0.5]], dtype=torch.double)

    # Without including 96 in failure_codes, status 96 is treated as success
    y_no_flag = obj_mod.J(
        x,
        parameters=[_LIN_KEY],
        true_observables={'R_obs': 1.0},
        worker=0,
        iter=0,
        ref_config='ref.toml',
        output='out',
        failure_codes=[],
    )
    assert y_no_flag.item() != pytest.approx(obj_mod.BAD_OBJ_VALUE)

    # With 96 in failure_codes, J must return BAD_OBJ_VALUE
    y_flagged = obj_mod.J(
        x,
        parameters=[_LIN_KEY],
        true_observables={'R_obs': 1.0},
        worker=0,
        iter=0,
        ref_config='ref.toml',
        output='out',
        failure_codes=[96],
    )
    assert y_flagged.item() == pytest.approx(obj_mod.BAD_OBJ_VALUE, rel=1e-9)
