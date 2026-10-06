"""
Tests for the inference pipeline entrypoints.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest
import toml

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.inference as inference_mod  # noqa: E402
from proteus.config import UnknownConfigKeyError  # noqa: E402
from proteus.inference.failures import ABORT_ON_FAILURE_ENV  # noqa: E402
from proteus.inference.objective import (  # noqa: E402
    _CHILD_TIMEOUT_ENV as CHILD_TIMEOUT_ENV,
)
from proteus.inference.objective import (  # noqa: E402
    SPECTRAL_CACHE_DIR,
    SPECTRAL_CACHE_ENV,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

BASE_CONFIG = str(Path(__file__).parent / 'base.toml')

# Pytest can hang on process completion when using multiprocessing by default.
mp.set_start_method('spawn', force=True)


@pytest.mark.unit
def test_run_inference_rejects_too_many_workers(monkeypatch, tmp_path):
    """``run_inference`` rejects ``n_workers >= cpu_count`` with a
    'Not enough CPU cores' error, so a misconfigured job fails at
    config-load rather than DoSing the host machine.
    """
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 4,
        'ref_config': 'tests/inference/base.toml',
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'P_surf': 1.0},
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
    }
    output_root = tmp_path / 'output'

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': str(tmp_path)},
    )
    monkeypatch.setattr(inference_mod, 'safe_rm', lambda _path: None)
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod, 'str_time', lambda: '2026-04-30 00:00:00 UTC')
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 4)

    create_init_calls: list = []
    monkeypatch.setattr(
        inference_mod, 'create_init', lambda *a, **kw: create_init_calls.append((a, kw))
    )
    with pytest.raises(RuntimeError, match='Not enough CPU cores'):
        inference_mod.run_inference(config)
    # Discrimination: the CPU-count guard fires before initial sampling. A
    # guard that raised only at parallel_process would leave calls here.
    assert create_init_calls == []


@pytest.mark.unit
def test_run_inference_raises_for_missing_reference_config(monkeypatch, tmp_path):
    """``run_inference`` raises FileNotFoundError when ``ref_config`` does
    not point to an existing file on disk, naming the missing path.
    """
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 1,
        'ref_config': 'missing.toml',
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'P_surf': 1.0},
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
    }
    output_root = tmp_path / 'output'

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': str(tmp_path)},
    )
    monkeypatch.setattr(inference_mod, 'safe_rm', lambda _path: None)
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod, 'str_time', lambda: '2026-04-30 00:00:00 UTC')
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 8)
    monkeypatch.setattr(inference_mod.os.path, 'isfile', lambda _path: False)

    create_init_calls: list = []
    monkeypatch.setattr(
        inference_mod, 'create_init', lambda *a, **kw: create_init_calls.append((a, kw))
    )
    with pytest.raises(FileNotFoundError, match='Cannot find reference config'):
        inference_mod.run_inference(config)
    # Discrimination: the missing-config guard must fire before initial
    # design generation. A regression that built the init dataset and
    # only failed later would have a non-empty call list here.
    assert create_init_calls == []


@pytest.mark.unit
def test_infer_from_config_loads_toml_and_dispatches(monkeypatch, tmp_path):
    """``infer_from_config(path)`` parses the TOML and forwards the
    resulting dict verbatim to ``run_inference``; no field is dropped or
    renamed in the dispatch step.
    """
    config_path = tmp_path / 'inference.toml'
    expected = {'output': 'dummy', 'n_workers': 1}
    config_path.write_text(toml.dumps(expected), encoding='utf-8')

    observed = {}

    def fake_run_inference(cfg):
        observed['config'] = cfg

    monkeypatch.setattr(inference_mod, 'run_inference', fake_run_inference)

    inference_mod.infer_from_config(str(config_path))

    assert observed['config'] == expected
    # Discrimination: a regression that mutated the dispatched config in
    # place would leave a still-equal-to-expected dict but with extra keys
    # injected. Pin the exact key set.
    assert set(observed['config'].keys()) == set(expected.keys())


# ============================================================================
# Reference-config validation before any worker is launched
# ============================================================================


@pytest.mark.unit
def test_parameter_bounds_converts_pairs_and_rejects_malformed_ranges():
    """``parameter_bounds`` accepts an increasing pair of numbers and returns
    it as floats, and rejects every other shape a user could write: a single
    value, a non-numeric entry, a decreasing pair, and the degenerate pair
    where the two ends coincide and the parameter has no range to search.
    """
    converted = inference_mod.parameter_bounds(
        {'planet.mass_tot': [1, 3], 'interior_struct.core_frac': (0.3, 0.7)}
    )
    assert converted['planet.mass_tot'] == (pytest.approx(1.0), pytest.approx(3.0))
    # Integer TOML literals must arrive as floats, matching the values the
    # optimiser writes back into each worker's config.
    assert all(isinstance(v, float) for v in converted['planet.mass_tot'])
    assert converted['interior_struct.core_frac'] == (
        pytest.approx(0.3),
        pytest.approx(0.7),
    )

    with pytest.raises(ValueError, match='pair of numbers'):
        inference_mod.parameter_bounds({'planet.mass_tot': [1.0]})
    with pytest.raises(ValueError, match='pair of numbers'):
        inference_mod.parameter_bounds({'planet.mass_tot': 'auto'})
    with pytest.raises(ValueError, match='must increase'):
        inference_mod.parameter_bounds({'planet.mass_tot': [3.0, 1.0]})
    # Edge case: coincident bounds are a zero-width range, not a fixed value.
    with pytest.raises(ValueError, match='must increase'):
        inference_mod.parameter_bounds({'planet.mass_tot': [2.0, 2.0]})
    # Edge case: TOML admits `inf`, which satisfies "increases" and clears the
    # schema's own range checks, then makes every unnormalised sample infinite.
    with pytest.raises(ValueError, match='must be finite'):
        inference_mod.parameter_bounds({'planet.mass_tot': [1.0, float('inf')]})
    with pytest.raises(ValueError, match='must be finite'):
        inference_mod.parameter_bounds({'planet.mass_tot': [float('nan'), 3.0]})


@pytest.mark.unit
def test_parameter_bounds_rejects_a_log_scaled_range_that_reaches_zero():
    """A parameter swept on a log scale cannot have a bound at or below zero:
    the optimiser samples it in log10 space. The range is rejected here, while
    the config is being read, not in every worker after the previous study
    has been removed. `planet.elements.H_budget` is log-scaled; a linear
    parameter with the same bounds is accepted.
    """
    with pytest.raises(ValueError, match='log scale') as excinfo:
        inference_mod.parameter_bounds({'planet.elements.H_budget': [0.0, 2e4]})
    # The message names the offending parameter and the range as written.
    assert "'planet.elements.H_budget'" in str(excinfo.value)
    assert '[0, 20000]' in str(excinfo.value)
    with pytest.raises(ValueError, match='log scale'):
        inference_mod.parameter_bounds({'planet.elements.H_budget': [-10.0, 2e4]})
    # A range that is negative throughout is rejected on the same grounds.
    with pytest.raises(ValueError, match='log scale'):
        inference_mod.parameter_bounds({'orbit.semimajoraxis': [-2.0, -1.0]})

    # Edge case: the smallest positive lower bound is accepted, so the check is
    # `> 0` and not a threshold further from zero.
    tiny = inference_mod.parameter_bounds({'planet.elements.H_budget': [5e-324, 2e4]})
    assert tiny['planet.elements.H_budget'][0] > 0.0

    # Discrimination: the same bounds on a linear parameter are valid, so the
    # rejection comes from the log scale, not from the value itself.
    linear = inference_mod.parameter_bounds({'outgas.fO2_shift_IW': [-4.0, 0.0]})
    assert linear['outgas.fO2_shift_IW'] == (pytest.approx(-4.0), pytest.approx(0.0))
    assert inference_mod.variable_is_logarithmic('outgas.fO2_shift_IW') is False
    assert inference_mod.variable_is_logarithmic('planet.elements.H_budget') is True


@pytest.mark.unit
@pytest.mark.parametrize('enabled', [True, False], ids=['cache-on', 'cache-off'])
def test_validate_reference_config_checks_the_spectral_cache_the_workers_use(
    monkeypatch, tmp_path, enabled
):
    """The configs checked at startup carry the spectral cache the workers will
    run with, which the inference switch sets whatever the reference config
    holds. The reference config here sets a path of its own to show that.
    """
    ref = toml.load(BASE_CONFIG)
    ref.setdefault('atmos_clim', {})['spectral_cache'] = '/my/cache'
    ref_config = tmp_path / 'ref.toml'
    ref_config.write_text(toml.dumps(ref), encoding='utf-8')

    checked = []
    monkeypatch.setattr(
        inference_mod, '_reject_bad_config', lambda raw, label: checked.append((label, raw))
    )
    inference_mod.validate_reference_config(
        str(ref_config), {'planet.mass_tot': [0.7, 3.0]}, spectral_cache=enabled
    )

    # The file as written, then both bound variants.
    assert len(checked) == 3
    assert checked[0][1]['atmos_clim']['spectral_cache'] == '/my/cache'
    variants = [raw['atmos_clim']['spectral_cache'] for _label, raw in checked[1:]]
    if enabled:
        assert all(v.endswith(SPECTRAL_CACHE_DIR) for v in variants)
    else:
        assert variants == ['none', 'none']


@pytest.mark.unit
def test_validate_reference_config_accepts_a_runnable_sweep():
    """A reference config that PROTEUS accepts, swept over parameters that stay
    inside the schema at both ends, passes validation. Each accepted sweep is
    paired with a neighbouring rejected one, so a validator gutted to an
    immediate return fails this test rather than passing it.
    """
    bounds = {'planet.mass_tot': [0.7, 3.0], 'interior_struct.core_frac': [0.3, 0.9]}
    inference_mod.validate_reference_config(BASE_CONFIG, bounds)
    # Liveness: widening one range past the schema limit must be refused, which
    # proves the accepted case above was actually checked.
    with pytest.raises(ValueError) as excinfo:
        inference_mod.validate_reference_config(
            BASE_CONFIG, {**bounds, 'interior_struct.core_frac': [0.3, 1.5]}
        )
    assert 'core_frac' in str(excinfo.value)
    # Only the widened range is at fault; the untouched one must not be named.
    assert 'mass_tot' not in str(excinfo.value)

    # Edge case: an empty sweep still validates the file itself, so a broken
    # reference config is caught even when nothing is being optimised.
    inference_mod.validate_reference_config(BASE_CONFIG, {})


@pytest.mark.unit
def test_validate_reference_config_rejects_a_mistyped_parameter_name():
    """A parameter name that no config field matches is reported as an
    unrecognised key. Without this check the name would be written into each
    worker's config as a new orphan section and every worker would refuse to
    start, midway through the study.
    """
    bounds = {'planet.mass_tott': [0.7, 3.0]}
    with pytest.raises(UnknownConfigKeyError) as excinfo:
        inference_mod.validate_reference_config(BASE_CONFIG, bounds)

    message = str(excinfo.value)
    assert 'planet.mass_tott' in message
    # The key is absent from the file on disk, so the message must say the
    # sweep introduced it rather than blaming the reference config alone.
    assert 'bounds' in message
    # Discrimination: the correctly spelled name must not be flagged, which
    # would happen if the walk reported every swept key rather than orphans.
    assert 'planet.mass_tot"' not in message


@pytest.mark.unit
def test_validate_reference_config_rejects_a_bound_outside_the_schema_range():
    """A range whose upper end leaves the interval the schema allows is
    rejected, and the message names the end that failed. ``core_frac`` is
    constrained to the open interval (0, 1), so 0.3 is accepted and 1.5 is
    not; only a check at both ends of the range catches this.
    """
    with pytest.raises(ValueError, match='upper bounds'):
        inference_mod.validate_reference_config(
            BASE_CONFIG, {'interior_struct.core_frac': [0.3, 1.5]}
        )
    # The same fault at the other end is attributed to the other end.
    with pytest.raises(ValueError, match='lower bounds'):
        inference_mod.validate_reference_config(
            BASE_CONFIG, {'interior_struct.core_frac': [-0.2, 0.9]}
        )
    # Discrimination: a range wholly inside (0, 1) must pass, so the failures
    # above come from the bounds and not from the reference config itself.
    inference_mod.validate_reference_config(
        BASE_CONFIG, {'interior_struct.core_frac': [0.3, 0.9]}
    )


@pytest.mark.unit
def test_validate_reference_config_rejects_a_faulty_reference_file(tmp_path):
    """A fault in the reference config itself is attributed to the file, not
    to the parameter sweep, so the user knows which file to edit.
    """
    raw = toml.load(BASE_CONFIG)
    raw['planet']['mass_tott'] = 1.0
    faulty = tmp_path / 'faulty.toml'
    faulty.write_text(toml.dumps(raw), encoding='utf-8')

    with pytest.raises(UnknownConfigKeyError) as excinfo:
        inference_mod.validate_reference_config(str(faulty), {'planet.mass_tot': [0.7, 3.0]})

    message = str(excinfo.value)
    assert 'planet.mass_tott' in message
    # Attribution: the file is named without the bounds qualifier, which is
    # only appended when the sweep is what introduced the fault.
    assert f'in {faulty}:' in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ('parameters', 'extra', 'error', 'match'),
    [
        ({'planet.mass_tott': [0.7, 3.0]}, {}, UnknownConfigKeyError, 'planet.mass_tott'),
        # Accepted by the schema, and rejected by the optimiser only once the
        # workers start sampling it in log10 space.
        ({'planet.elements.H_budget': [0.0, 2e4]}, {}, ValueError, 'log scale'),
        # A quoted "false" is truthy, so it would silently leave the cache on.
        ({'planet.mass_tot': [0.7, 3.0]}, {'spectral_cache': 'false'}, ValueError, 'true or'),
    ],
    ids=['misspelt_parameter', 'log_scaled_range_reaching_zero', 'spectral_cache_not_bool'],
)
def test_run_inference_validates_reference_config_before_emptying_output(
    monkeypatch, tmp_path, parameters, extra, error, match
):
    """``run_inference`` validates the reference config before it empties the
    study output folder and before it generates any initial design. Re-running
    a finished study with a typo'd parameter name, or with a range the
    optimiser cannot sample, must cost the user neither simulation time nor
    the previous study's results.
    """
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 1,
        'ref_config': BASE_CONFIG,
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'P_surf': 1.0},
        'parameters': parameters,
        **extra,
    }
    # Stand in for a completed earlier study occupying the same output folder.
    output_root = tmp_path / 'output'
    output_root.mkdir()
    previous = output_root / 'init.csv'
    previous.write_text('x_0,y\n0.5,1.0\n', encoding='utf-8')

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': ''},
    )
    # `safe_rm` is deliberately left real: the point of the test is that it
    # never runs.
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod, 'str_time', lambda: '2026-04-30 00:00:00 UTC')
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 8)

    create_init_calls: list = []
    monkeypatch.setattr(
        inference_mod, 'create_init', lambda *a, **kw: create_init_calls.append((a, kw))
    )

    with pytest.raises(error, match=match):
        inference_mod.run_inference(config)
    # Ordering: the guard must fire before the initial design is generated.
    assert create_init_calls == []
    # ...and before the output folder is emptied. A guard placed after the
    # `safe_rm` call would leave this file deleted.
    assert previous.read_text(encoding='utf-8') == 'x_0,y\n0.5,1.0\n'
    # Nothing downstream of the guard may have run at all.
    assert not (output_root / 'ref_config.toml').exists()


class _StopAfterSetup(Exception):
    """Raised in place of the initial design, once startup has finished."""


@pytest.mark.unit
@pytest.mark.parametrize('switch', [None, True, False], ids=['default', 'on', 'off'])
def test_run_inference_reports_the_spectral_cache_the_study_uses(
    monkeypatch, tmp_path, caplog, switch
):
    """The study log names the spectral cache the workers will use, so a study
    running without one is visible. The cache is on unless the inference config
    turns it off, and the setting reaches the workers through the environment.
    """
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 1,
        'ref_config': BASE_CONFIG,
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'P_surf': 1.0},
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
    }
    if switch is not None:
        config['spectral_cache'] = switch
    output_root = tmp_path / 'output'
    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': ''},
    )
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 8)

    def _stop(*_a, **_kw):
        raise _StopAfterSetup

    monkeypatch.setattr(inference_mod, 'create_init', _stop)
    # Registered so the values run_inference writes are undone afterwards.
    for env in (SPECTRAL_CACHE_ENV, ABORT_ON_FAILURE_ENV, CHILD_TIMEOUT_ENV):
        monkeypatch.setenv(env, 'unset')

    with caplog.at_level('INFO'), pytest.raises(_StopAfterSetup):
        inference_mod.run_inference(config)

    lines = [r.getMessage() for r in caplog.records if 'Spectral cache' in r.getMessage()]
    assert len(lines) == 1
    if switch is False:
        assert (
            lines[0] == 'Spectral cache: off (spectral_cache = false in the inference config)'
        )
        assert os.environ[SPECTRAL_CACHE_ENV] == '0'
    else:
        assert lines[0].endswith(f'unit_inference/{SPECTRAL_CACHE_DIR}')
        assert os.environ[SPECTRAL_CACHE_ENV] == '1'


@pytest.mark.unit
def test_run_inference_rejects_incomplete_sigma_before_emptying_output(monkeypatch, tmp_path):
    """A ``[sigma]`` table missing an observable is refused before the output
    folder is emptied, so a typo does not cost the previous study's results.
    A complete table is stored back on the config as floats.
    """
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 1,
        'ref_config': BASE_CONFIG,
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'R_obs': 6.0e6, 'T_obs': 400.0},
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
        'sigma': {'R_obs': 1.0e5},
    }
    output_root = tmp_path / 'output'
    output_root.mkdir()
    previous = output_root / 'init.csv'
    previous.write_text('x_0,y\n0.5,1.0\n', encoding='utf-8')

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': ''},
    )
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 8)
    create_init_calls: list = []
    monkeypatch.setattr(
        inference_mod, 'create_init', lambda *a, **kw: create_init_calls.append((a, kw))
    )

    with pytest.raises(ValueError, match='T_obs'):
        inference_mod.run_inference(config)
    assert create_init_calls == []
    # A check placed after `safe_rm` would leave this file deleted.
    assert previous.read_text(encoding='utf-8') == 'x_0,y\n0.5,1.0\n'

    # Complete table: accepted, and the run proceeds to the initial design.
    # Stopped there, since only the validation step is under test.
    config['sigma'] = {'T_obs': 20, 'R_obs': 1.0e5}
    # run_inference records these for its workers; restored after the test.
    for name in (
        'PROTEUS_INFERENCE_CHILD_TIMEOUT_S',
        'PROTEUS_INFERENCE_DISPATCH',
        'PROTEUS_INFERENCE_RUNNER_MAX_JOBS',
        inference_mod.ABORT_ON_FAILURE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    def _stop(cfg):
        raise RuntimeError('stop after validation')

    monkeypatch.setattr(inference_mod, 'create_init', _stop)
    with pytest.raises(RuntimeError, match='stop after validation'):
        inference_mod.run_inference(config)
    assert config['sigma'] == {'T_obs': pytest.approx(20.0), 'R_obs': pytest.approx(1.0e5)}
    assert isinstance(config['sigma']['T_obs'], float)


@pytest.mark.unit
def test_run_inference_rejects_invalid_correlation_before_emptying_output(
    monkeypatch, tmp_path
):
    """An invalid ``[correlation]`` table is refused before the output folder is
    emptied; a valid one is stored back as floats.
    """
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 1,
        'ref_config': BASE_CONFIG,
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'R_obs': 6.0e6, 'T_obs': 400.0, 'g_obs': 9.8},
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
        'correlation': {'R_obs': {'T_obs': 0.5}},
    }
    output_root = tmp_path / 'output'
    output_root.mkdir()
    previous = output_root / 'init.csv'
    previous.write_text('x_0,y\n0.5,1.0\n', encoding='utf-8')

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': ''},
    )
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 8)
    for name in (
        'PROTEUS_INFERENCE_CHILD_TIMEOUT_S',
        'PROTEUS_INFERENCE_DISPATCH',
        'PROTEUS_INFERENCE_RUNNER_MAX_JOBS',
        inference_mod.ABORT_ON_FAILURE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    def _stop(cfg):
        raise RuntimeError('stop after validation')

    monkeypatch.setattr(inference_mod, 'create_init', _stop)

    with pytest.raises(ValueError, match='sigma'):
        inference_mod.run_inference(dict(config))
    config['sigma'] = {'R_obs': 1.0e5, 'T_obs': 20.0, 'g_obs': 0.5}
    # Pairwise valid, but smallest eigenvalue 1 - 1.8 < 0.
    bad = {'R_obs': {'T_obs': 0.9, 'g_obs': 0.9}, 'T_obs': {'g_obs': -0.9}}
    with pytest.raises(ValueError, match='positive definite'):
        inference_mod.run_inference({**config, 'correlation': bad})
    # A check placed after `safe_rm` would leave this file deleted.
    assert previous.read_text(encoding='utf-8') == 'x_0,y\n0.5,1.0\n'

    config['correlation'] = {'R_obs': {'T_obs': 1}}
    with pytest.raises(ValueError, match=r'\(-1, 1\)'):
        inference_mod.run_inference(dict(config))
    config['correlation'] = {'R_obs': {'T_obs': -0.25}}
    with pytest.raises(RuntimeError, match='stop after validation'):
        inference_mod.run_inference(config)
    assert config['correlation'] == {'R_obs': {'T_obs': pytest.approx(-0.25)}}


@pytest.mark.unit
def test_validate_truth_returns_ordered_floats_and_rejects_bad_tables():
    """A complete ``[truth]`` table comes back as floats in parameter order;
    a missing, unknown, non-finite, boolean or non-positive log-scaled entry
    is refused with the offending name in the message.
    """
    pars = {
        'outgas.fO2_shift_IW': [-4.0, 4.0],
        'planet.elements.H_budget': [1e3, 2e4],
    }
    # Written in reverse order and with an int, so both conversions are exercised.
    out = inference_mod.validate_truth(
        pars, {'planet.elements.H_budget': 5000, 'outgas.fO2_shift_IW': -1.5}
    )
    assert list(out) == list(pars)
    assert out['planet.elements.H_budget'] == pytest.approx(5.0e3, rel=1e-12)
    assert isinstance(out['planet.elements.H_budget'], float)
    # Edge case: a zero or negative true value is valid for a linear parameter.
    assert out['outgas.fO2_shift_IW'] == pytest.approx(-1.5, rel=1e-12)
    assert inference_mod.validate_truth(pars, None) is None

    with pytest.raises(ValueError, match='H_budget'):
        inference_mod.validate_truth(pars, {'outgas.fO2_shift_IW': 0.0})
    with pytest.raises(ValueError, match='not parameters.*mass_tot'):
        inference_mod.validate_truth(
            pars,
            {
                'outgas.fO2_shift_IW': 0.0,
                'planet.elements.H_budget': 5e3,
                'planet.mass_tot': 1.0,
            },
        )
    with pytest.raises(ValueError, match='finite number'):
        inference_mod.validate_truth(
            pars, {'outgas.fO2_shift_IW': float('nan'), 'planet.elements.H_budget': 5e3}
        )
    with pytest.raises(ValueError, match='finite number'):
        inference_mod.validate_truth(
            pars, {'outgas.fO2_shift_IW': True, 'planet.elements.H_budget': 5e3}
        )
    # H_budget is sampled in log10, where a non-positive value has no position.
    with pytest.raises(ValueError, match='log space'):
        inference_mod.validate_truth(
            pars, {'outgas.fO2_shift_IW': 0.0, 'planet.elements.H_budget': 0.0}
        )


@pytest.mark.unit
def test_truth_outside_bounds_flags_only_values_beyond_the_range():
    """True values on a bound are recoverable and not flagged; values beyond
    either bound are returned by name so the study can warn about them.
    """
    pars = {'a.x': [0.0, 1.0], 'a.y': [2.0, 3.0], 'a.z': [-1.0, 1.0]}
    # Edge case: 'a.x' sits exactly on its lower bound.
    truth = {'a.x': 0.0, 'a.y': 3.5, 'a.z': -1.2}

    out = inference_mod.truth_outside_bounds(pars, truth)

    assert set(out) == {'a.y', 'a.z'}
    assert out['a.y'] == pytest.approx(3.5, rel=1e-12)
    assert inference_mod.truth_outside_bounds(pars, None) == {}


@pytest.mark.unit
def test_run_inference_rejects_incomplete_truth_before_emptying_output(monkeypatch, tmp_path):
    """A ``[truth]`` table missing a parameter is refused before the output
    folder is emptied, and a complete one is stored back as floats.
    """
    config = {
        'output': 'unit_inference',
        'logging': 'INFO',
        'n_workers': 1,
        'ref_config': BASE_CONFIG,
        'n_steps': 1,
        'kernel': 'MAT3/2',
        'acqf': 'LogEI',
        'seed': 1,
        'observables': {'R_obs': 6.0e6},
        'parameters': {'planet.mass_tot': [0.7, 3.0], 'interior_struct.core_frac': [0.3, 0.7]},
        'truth': {'planet.mass_tot': 1.0},
    }
    output_root = tmp_path / 'output'
    output_root.mkdir()
    previous = output_root / 'init.csv'
    previous.write_text('x_0,y\n0.5,1.0\n', encoding='utf-8')

    monkeypatch.setattr(
        inference_mod,
        'get_proteus_directories',
        lambda _output: {'output': str(output_root), 'proteus': ''},
    )
    monkeypatch.setattr(inference_mod, 'setup_logger', lambda **_kwargs: None)
    monkeypatch.setattr(inference_mod.os, 'cpu_count', lambda: 8)

    with pytest.raises(ValueError, match='core_frac'):
        inference_mod.run_inference(config)
    # A check placed after `safe_rm` would leave this file deleted.
    assert previous.read_text(encoding='utf-8') == 'x_0,y\n0.5,1.0\n'

    config['truth'] = {'planet.mass_tot': 1, 'interior_struct.core_frac': 0.325}
    for name in (
        'PROTEUS_INFERENCE_CHILD_TIMEOUT_S',
        'PROTEUS_INFERENCE_DISPATCH',
        'PROTEUS_INFERENCE_RUNNER_MAX_JOBS',
        inference_mod.ABORT_ON_FAILURE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    def _stop(cfg):
        raise RuntimeError('stop after validation')

    monkeypatch.setattr(inference_mod, 'create_init', _stop)
    with pytest.raises(RuntimeError, match='stop after validation'):
        inference_mod.run_inference(config)
    assert config['truth']['planet.mass_tot'] == pytest.approx(1.0, rel=1e-12)
    assert isinstance(config['truth']['planet.mass_tot'], float)


# ============================================================================
# Regression: no stray prints + docstring uses current schema
# ============================================================================


@pytest.mark.unit
def test_infer_from_config_uses_logger_not_print(caplog, tmp_path, monkeypatch):
    """Regression: infer_from_config must route its startup message through
    the module logger, not print(). The original PR #675 BayesOpt rewrite
    migrated every other print() in the inference src to log.info; this
    one was missed and is corrected here."""
    import io
    from contextlib import redirect_stdout

    import proteus.inference.inference as inference_mod

    # Stub the heavy run path; we only want the early log statement.
    monkeypatch.setattr(inference_mod, 'run_inference', lambda _cfg: None)

    cfg_path = tmp_path / 'dummy.infer.toml'
    cfg_path.write_text('seed = 1\noutput = "x"\nlogging = "INFO"\n', encoding='utf-8')

    buf = io.StringIO()
    with caplog.at_level('INFO', logger='fwl.proteus.inference.inference'):
        with redirect_stdout(buf):
            inference_mod.infer_from_config(str(cfg_path))

    # Logger captured the message...
    assert any('Inference config:' in rec.message for rec in caplog.records), (
        'infer_from_config must emit Inference config via log.info'
    )
    # ...and stdout did not.
    assert 'Inference config:' not in buf.getvalue(), (
        'infer_from_config must not write to stdout via print()'
    )


@pytest.mark.unit
def test_get_nested_docstring_uses_current_schema_example():
    """Regression: utils.get_nested docstring example must use a
    parameter path that is valid on the current branch schema
    (planet.mass_tot). The deprecated struct.mass_tot path must not
    appear; it would mislead any reader who copies the docstring example
    into a working config."""
    from inspect import getdoc

    from proteus.inference.utils import get_nested

    doc = getdoc(get_nested)
    assert doc is not None
    assert 'struct.mass_tot' not in doc, (
        'docstring example must not reference the deprecated struct.* schema'
    )
    assert 'planet.mass_tot' in doc, 'docstring example must use the current planet.* schema'
