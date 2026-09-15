"""
Tests for the inference pipeline entrypoints.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import multiprocessing as mp
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
    # Discrimination: the CPU-count guard must fire before any expensive
    # initial-design dispatch. A regression that allowed init sampling to
    # start and only raised at parallel_process would have a non-empty
    # call list here.
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
def test_run_inference_validates_reference_config_before_emptying_output(monkeypatch, tmp_path):
    """``run_inference`` validates the reference config before it empties the
    study output folder and before it generates any initial design. Re-running
    a finished study with a typo'd parameter name must cost the user neither
    simulation time nor the previous study's results.
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
        'parameters': {'planet.mass_tott': [0.7, 3.0]},
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

    with pytest.raises(UnknownConfigKeyError, match='planet.mass_tott'):
        inference_mod.run_inference(config)
    # Ordering: the guard must fire before the initial design is generated.
    assert create_init_calls == []
    # ...and before the output folder is emptied. A guard placed after the
    # `safe_rm` call would leave this file deleted.
    assert previous.read_text(encoding='utf-8') == 'x_0,y\n0.5,1.0\n'
    # Nothing downstream of the guard may have run at all.
    assert not (output_root / 'ref_config.toml').exists()


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
