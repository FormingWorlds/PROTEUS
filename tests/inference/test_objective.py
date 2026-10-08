"""
Unit tests for inference objective helpers and simulator wrapping.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import csv
import functools
import logging
import math
import subprocess

import pandas as pd
import pytest
import toml

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
torch = pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.failures as failures_mod  # noqa: E402
import proteus.inference.objective as objective_mod  # noqa: E402
from proteus.inference.likelihood import CorrelationWhitener  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_log_warp_monotonic_decreasing_in_squared_distance():
    """``log_warp(sq_dist)`` returns -log10(sq_dist + 1e-10): values
    closer to the target (sq_dist near 0) score higher than distant
    ones. Discrimination: a regression that flipped the sign would
    invert the ranking; a regression that dropped the offset 1e-10
    would diverge at sq_dist=0.
    """
    near = torch.tensor([1e-4], dtype=torch.double)
    far = torch.tensor([1.0], dtype=torch.double)
    score_near = objective_mod.log_warp(near)
    score_far = objective_mod.log_warp(far)
    # Closer (smaller sq_dist) -> larger score
    assert score_near.item() > score_far.item()
    # Scale guards: -log10(1e-4) ~ 4, -log10(1.0) ~ 0
    assert 3 < score_near.item() < 5
    assert -0.5 < score_far.item() < 0.5


def test_log_warp_finite_at_exact_zero():
    """``log_warp(0.0)`` does not diverge: the 1e-10 offset guarantees
    finite output. Discrimination: a regression that removed the offset
    would emit -inf, which would NaN-poison downstream BO maths.
    """
    val = objective_mod.log_warp(torch.tensor([0.0], dtype=torch.double))
    assert torch.isfinite(val).all()
    # Expected ~ -log10(1e-10) = 10
    assert 9 < val.item() < 11


def test_update_toml_updates_nested_keys(tmp_path):
    """``update_toml`` applies dotted-key overrides on the loaded config
    (e.g. ``section.value=2``) and creates intermediate nesting for keys
    that did not exist in the base (``new.branch.leaf=3``).
    """
    base_cfg = {'section': {'value': 1}}
    config_file = tmp_path / 'base.toml'
    out_file = tmp_path / 'nested' / 'updated.toml'
    config_file.write_text(toml.dumps(base_cfg), encoding='utf-8')

    objective_mod.update_toml(
        str(config_file),
        {'section.value': 2, 'new.branch.leaf': 3},
        str(out_file),
    )

    loaded = toml.loads(out_file.read_text(encoding='utf-8'))
    assert loaded['section']['value'] == 2
    assert loaded['new']['branch']['leaf'] == 3


@pytest.mark.parametrize('enabled', [True, False], ids=['cache-on', 'cache-off'])
@pytest.mark.parametrize(
    'ref_cache', [None, 'none', '/my/cache'], ids=['unset', 'none', 'path']
)
def test_run_proteus_takes_the_spectral_cache_from_the_inference_switch(
    monkeypatch, tmp_path, enabled, ref_cache
):
    """The inference config's `spectral_cache` switch decides the cache every
    worker runs with, whatever the reference config sets: a reference config
    copied from `all_options.toml` carries "none" without the user choosing it.
    On, workers share the study's cache folder; off, they run with "none".
    Both config writes, before and after the run, carry the same value.
    """
    monkeypatch.setenv('PROTEUS_OUTPUT_PATH', str(tmp_path))
    monkeypatch.setenv(objective_mod.SPECTRAL_CACHE_ENV, '1' if enabled else '0')
    ref = {'planet': {'mass_tot': 1.0}}
    if ref_cache is not None:
        ref['atmos_clim'] = {'spectral_cache': ref_cache}
    ref_config = tmp_path / 'reference.toml'
    ref_config.write_text(toml.dumps(ref), encoding='utf-8')
    run_dir = tmp_path / 'study' / 'workers' / 'w_0' / 'i_0'

    seen = []

    def _fake_run(command, **_kwargs):
        seen.append(toml.load(command[3])['atmos_clim']['spectral_cache'])
        pd.DataFrame([{'P_surf': 1e5}]).to_csv(
            run_dir / 'runtime_helpfile.csv', sep=' ', index=False
        )

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)
    objective_mod.run_proteus(
        parameters={'planet.mass_tot': 2.0},
        worker=0,
        iter=0,
        observables=['P_surf'],
        ref_config=str(ref_config),
        output='study',
    )
    written = toml.load(run_dir / 'input.toml')
    study_cache = str(tmp_path / 'study' / objective_mod.SPECTRAL_CACHE_DIR)
    assert seen == [study_cache if enabled else 'none']
    assert written['atmos_clim']['spectral_cache'] == seen[0]
    # The swept value is still applied alongside.
    assert written['planet']['mass_tot'] == pytest.approx(2.0)


def test_spectral_cache_switch_defaults_to_on(monkeypatch):
    """Workers started without the switch recorded share the cache, and only
    an explicit "0" turns it off.
    """
    monkeypatch.delenv(objective_mod.SPECTRAL_CACHE_ENV, raising=False)
    assert objective_mod.spectral_cache_enabled() is True
    monkeypatch.setenv(objective_mod.SPECTRAL_CACHE_ENV, '0')
    assert objective_mod.spectral_cache_enabled() is False
    monkeypatch.setenv(objective_mod.SPECTRAL_CACHE_ENV, '1')
    assert objective_mod.spectral_cache_enabled() is True
    # Off gives the literal the config converter reads as disabled.
    assert objective_mod.worker_spectral_cache('study', False) == 'none'


def test_apply_nested_updates_mutates_in_place_and_rejects_value_paths():
    """``apply_nested_updates`` writes dotted keys into the dict it was given,
    creating the sections a new key needs, and refuses a path that descends
    through an entry holding a value. The refusal matters because a swept
    parameter name is user-supplied: ``planet.mass_tot.value`` would otherwise
    fail with an attribute error naming nothing the user wrote.
    """
    config = {'section': {'value': 1}}
    returned = objective_mod.apply_nested_updates(
        config, {'section.value': 2, 'new.branch.leaf': 3}
    )
    assert config['section']['value'] == 2
    assert config['new']['branch']['leaf'] == 3
    # In-place: the same object is handed back, so a caller holding the
    # original reference sees the updates.
    assert returned is config

    with pytest.raises(ValueError, match="'section.value' holds a value"):
        objective_mod.apply_nested_updates(config, {'section.value.deeper': 4})
    # The refused key is not written. Updates are applied as they are walked,
    # so an earlier key in the same call would already have been applied; this
    # pins only that the rejected one was not.
    assert config['section']['value'] == 2


def test_run_proteus_success_handles_escaped_atmosphere(monkeypatch, tmp_path):
    """``run_proteus`` handles the escaped-atmosphere case (P_surf=0):
    the observable dictionary is populated with zeros instead of NaN,
    and ``update_toml`` is invoked exactly twice (once per simulator pass)
    so the inversion harness sees a numeric value. The per-run config entries
    are added to what is written, never to the parameters passed in.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    pd.DataFrame([{'P_surf': 0.0, 'atm_kg_per_mol': 44.0}]).to_csv(
        out_abs / 'runtime_helpfile.csv', sep=' ', index=False
    )

    updates = []
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(
        objective_mod,
        'update_toml',
        lambda config_file, values, output_file: updates.append(
            (config_file, values, output_file)
        ),
    )
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)

    parameters = {'planet.mass_tot': 2.0}
    obs, status = objective_mod.run_proteus(
        parameters=parameters,
        worker=1,
        iter=2,
        observables=['P_surf', 'atm_kg_per_mol'],
        ref_config='reference.toml',
        output='dummy_output',
    )

    assert obs['P_surf'] == pytest.approx(0.0)
    assert obs['atm_kg_per_mol'] == pytest.approx(0.0)
    assert len(updates) == 2
    # The fixed entries reach the config that is written, but not the caller's
    # dict: `J` reuses that dict for the failure report, which formats every
    # value as a number and names only the swept parameters.
    assert updates[0][1]['params.out.path'] == 'dummy_output/workers/w_1/i_2'
    assert updates[0][1]['params.out.plot_mod'] == 'none'
    assert updates[0][1]['atmos_clim.spectral_cache'].endswith('spectral_cache')
    assert list(parameters) == ['planet.mass_tot']
    assert parameters['planet.mass_tot'] == pytest.approx(2.0)
    # No status file was written, which is reported as such rather than as a
    # generic error: a run that dies during start-up and a run that reaches
    # the main loop and fails there call for different investigations.
    assert status == objective_mod.STATUS_MISSING
    assert status != 20


def test_run_proteus_raises_when_command_missing(monkeypatch, tmp_path):
    """A ``FileNotFoundError`` from ``subprocess.run`` (i.e. the proteus
    binary is not on PATH) is wrapped as ``RuntimeError`` with a
    'command not found' message.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)

    run_calls = []

    def _fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        raise FileNotFoundError('missing')

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)

    with pytest.raises(RuntimeError, match='command not found') as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    # Cause guard: the FileNotFoundError is chained via __cause__. A bare
    # RuntimeError would match the text but lose the traceback.
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    # Side-effect guard: subprocess.run must have been invoked exactly
    # once. A regression that short-circuited before dispatch would
    # still raise but with a different (constant) error path.
    assert len(run_calls) == 1


def test_run_proteus_raises_when_command_fails(monkeypatch, tmp_path):
    """A non-zero exit from the proteus binary is reported as a
    ``ProteusRunFailure`` naming the run, its exit code, and the status the
    simulator recorded for itself, so the failure mode can be diagnosed
    without opening the study by hand.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    # The simulator recorded an atmosphere-model error before exiting. The
    # report must carry this, not a code inferred from the exit status.
    (out_abs / 'status').write_text('22\nError (Atmosphere model)\n', encoding='utf-8')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)

    def _fake_run(*_args, **kwargs):
        # The real simulator writes to the stream it is handed before it dies.
        kwargs['stdout'].write('boom\n')
        kwargs['stdout'].flush()
        raise subprocess.CalledProcessError(returncode=3, cmd=['proteus'])

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)

    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={'planet.mass_tot': 1.25},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    failure = excinfo.value
    # Cause-preservation guard: the original CalledProcessError must be
    # chained via __cause__ so the operator sees the failing command.
    assert isinstance(failure.__cause__, subprocess.CalledProcessError)
    # Exit-code-fidelity guard: a regression that always reported
    # 'exit code 0' or hardcoded a different code would still pass a
    # plain regex match if loose, so pin the integer through the cause.
    assert failure.__cause__.returncode == 3
    assert failure.exit_code == 3
    # Status fidelity: the status file is read on the failure path. A
    # regression that raised before reading it would report the missing
    # sentinel, and one that kept the old hard-coded fallback would report 20.
    assert failure.status == 22
    assert 'Atmosphere' in failure.status_desc
    # The swept parameter is named; the fixed per-run overrides are not,
    # because they carry no information about which sample failed.
    assert failure.parameters == {'planet.mass_tot': pytest.approx(1.25)}
    assert 'params.out.path' not in failure.parameters
    # The capture is kept beside the run folder, not inside it: the simulator
    # empties its own output directory once it starts, which would unlink a
    # file held open there.
    console = out_abs.parent / f'{out_abs.name}{objective_mod.CHILD_CONSOLE_SUFFIX}'
    assert console.is_file()
    assert 'boom' in console.read_text(encoding='utf-8')
    assert not (out_abs / console.name).exists()
    # The failure names that capture rather than copying its contents. A run
    # that dies before its own logger exists leaves nothing else to read, so a
    # report that named no path would leave the cause unreachable.
    assert failure.console_path == str(console)
    rendered = failure.report()
    assert 'worker=0 iter=0' in rendered
    assert 'planet.mass_tot=1.25' in rendered
    assert str(console) in rendered


def test_run_proteus_raises_on_missing_observable(monkeypatch, tmp_path):
    """Requesting an observable that the simulator did not write to the
    helpfile raises ``KeyError`` with a 'Requested observable' message,
    so a typo in the inference config fails loudly rather than producing
    silent NaN results.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    pd.DataFrame([{'P_surf': 1.0}]).to_csv(
        out_abs / 'runtime_helpfile.csv', sep=' ', index=False
    )

    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)

    with pytest.raises(KeyError, match='Requested observable') as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['not_present'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    # Identity guard: the KeyError names the missing observable. A generic
    # message would match the regex above but not say which field.
    assert 'not_present' in str(excinfo.value)
    # Discrimination: a valid observable on the same helpfile must
    # complete normally. This rules out a regression that hard-raises
    # KeyError on every input regardless of the observables list.
    obs, status = objective_mod.run_proteus(
        parameters={},
        worker=0,
        iter=0,
        observables=['P_surf'],
        ref_config='reference.toml',
        output='dummy_output',
    )
    assert obs['P_surf'] == pytest.approx(1.0)
    assert status == objective_mod.STATUS_MISSING


@pytest.mark.physics_invariant
def test_eval_obj_mixes_log_and_linear_variables(monkeypatch):
    """``eval_obj`` evaluates log-relative residuals for log-scaled
    observables and linear-relative residuals for linear ones, then
    returns ``-log10(sum_sq + 1e-10)``. The mixed-mode arithmetic is the
    point of this test: log and linear contributions enter the sum
    using different normalisations.
    """
    monkeypatch.setattr(objective_mod, 'variable_is_logarithmic', lambda key: key == 'P_surf')

    sim = {'P_surf': 1e-6, 'R_obs': 2.0}
    tru = {'P_surf': 1e-5, 'R_obs': 1.0}

    value = objective_mod.eval_obj(sim, tru)

    expected_sq = ((1.0 - (-6.0 / -5.0)) ** 2) + ((1.0 - 2.0 / 1.0) ** 2)
    expected = -torch.log10(torch.tensor([[expected_sq + 1e-10]], dtype=torch.double))
    assert value.item() == pytest.approx(expected.item())
    # Discrimination: treating P_surf as linear (1e-6 vs 1e-5, residual 0.9)
    # gives a very different objective from log mode (-6/-5 = 1.2, residual
    # 0.04). Pin against that wrong-mode value.
    sim_lin = {'P_surf': 1e-6, 'R_obs': 2.0}
    expected_sq_wrong = ((1.0 - 1e-6 / 1e-5) ** 2) + ((1.0 - 2.0 / 1.0) ** 2)
    expected_wrong = -torch.log10(
        torch.tensor([[expected_sq_wrong + 1e-10]], dtype=torch.double)
    )
    assert abs(value.item() - expected_wrong.item()) > 0.1
    # Boundedness guard: the objective is -log10(sum_sq + 1e-10), finite for
    # any sum_sq >= 0, so NaN or inf means a regression.
    assert torch.isfinite(value).all()
    # Identical sim == tru produces sum_sq = 0, hence -log10(1e-10) = 10.
    value_match = objective_mod.eval_obj(sim_lin, sim_lin)
    assert value_match.item() == pytest.approx(10.0, rel=1e-6)


def test_eval_obj_handles_zero_true_value():
    """Zero-valued observables should use an EPS_CLIP offset denominator
    to avoid division-by-zero.

    Discrimination: a denominator of exactly 0.0 would produce +inf or NaN;
    the EPS_CLIP offset must make the result finite.
    """
    sim = {'R_obs': 2.0}
    tru = {'R_obs': 0.0}

    value = objective_mod.eval_obj(sim, tru)

    denom = 0.0 + objective_mod.EPS_CLIP
    expected_sq = (1.0 - 2.0 / denom) ** 2
    expected = -torch.log10(
        torch.tensor([[expected_sq + objective_mod.EPS_CLIP]], dtype=torch.double)
    )
    assert value.item() == pytest.approx(expected.item())
    # Without EPS_CLIP the result would be +inf or NaN; finite output
    # is the key contract of the zero-denominator guard.
    assert torch.isfinite(value).all()


@pytest.mark.physics_invariant
def test_eval_obj_sigma_matched_by_key_not_order(monkeypatch):
    """With ``sigma``, each residual is divided by the uncertainty of the
    same observable, looked up by name, so a ``sigma`` dict listed in a
    different order than the observables gives the same chi-squared.
    A missing entry is rejected rather than silently misaligned.
    """
    monkeypatch.setattr(objective_mod, 'variable_is_logarithmic', lambda key: False)

    sim = {'R_obs': 7.0e6, 'T_obs': 450.0}
    tru = {'R_obs': 6.0e6, 'T_obs': 400.0}
    # Listed in the reverse order of `sim`. The two sigmas differ by 2e4, so
    # pairing them by position instead of by name changes the result.
    sigma = {'T_obs': 25.0, 'R_obs': 5.0e5}

    value = objective_mod.eval_obj(sim, tru, sigma)

    # (1e6 / 5e5)^2 + (50 / 25)^2 = 4 + 4 = 8
    expected_sq = 8.0
    assert value.item() == pytest.approx(-math.log10(expected_sq + 1e-10))
    # Order guard: pairing by position gives (1e6/25)^2 + (50/5e5)^2 = 1.6e9,
    # an objective about 8.3 lower.
    wrong_sq = (1.0e6 / 25.0) ** 2 + (50.0 / 5.0e5) ** 2
    assert abs(value.item() + math.log10(wrong_sq)) > 5.0
    # Relative-error guard: the default objective gives (1/6)^2 + (1/8)^2,
    # which differs from the chi-squared result.
    assert abs(value.item() - objective_mod.eval_obj(sim, tru).item()) > 1.0

    # Limit input: sim == tru gives zero chi-squared, hence -log10(1e-10) = 10.
    assert objective_mod.eval_obj(tru, tru, sigma).item() == pytest.approx(10.0, rel=1e-6)

    # Error contract: an observable without a sigma is named in the error.
    with pytest.raises(KeyError, match='T_obs'):
        objective_mod.eval_obj(sim, tru, {'R_obs': 5.0e5})


@pytest.mark.physics_invariant
def test_eval_obj_sigma_converted_to_dex_for_log_observables():
    """For an observable compared in log space, ``sigma`` is given in the
    observable's own units and converted to dex as sigma / (x ln 10). A
    model one sigma above the target then scores chi-squared of about one,
    as it does for a linear observable.
    """
    # atm_kg_per_mol is log-scaled, R_obs linear (utils.coupler).
    tru = {'atm_kg_per_mol': 0.02, 'R_obs': 6.0e6}
    sigma = {'atm_kg_per_mol': 0.002, 'R_obs': 1.0e5}
    # A factor-of-two miss on the log observable, sim exact on the linear one,
    # so the log term is the whole chi-squared.
    sim = {'atm_kg_per_mol': 0.04, 'R_obs': 6.0e6}

    value = objective_mod.eval_obj(sim, tru, sigma)

    sig_dex = 0.002 / (0.02 * math.log(10.0))
    expected_sq = (math.log10(2.0) / sig_dex) ** 2  # ~48.0
    assert value.item() == pytest.approx(-math.log10(expected_sq + 1e-10), rel=1e-9)
    # Unit guard: using sigma unconverted (as if already in dex) gives
    # (0.301 / 0.002)^2 ~ 2.3e4, an objective ~2.7 lower.
    wrong_raw = (math.log10(2.0) / 0.002) ** 2
    assert abs(value.item() + math.log10(wrong_raw)) > 2.0
    # Space guard: a linear comparison gives (0.02 / 0.002)^2 = 100, ~0.3 lower.
    wrong_lin = (0.02 / 0.002) ** 2
    assert abs(value.item() + math.log10(wrong_lin)) > 0.2

    # Small-sigma limit, where the first-order conversion is accurate: one
    # sigma above the target on each observable gives chi-squared of 2.
    small = {'atm_kg_per_mol': 2.0e-5, 'R_obs': 1.0e5}
    one_up = {k: tru[k] + small[k] for k in tru}
    chi2 = 10 ** (-objective_mod.eval_obj(one_up, tru, small).item())
    # Exact for R_obs; ln(1.001)/0.001 = 0.9995 for the log term, hence 2e-3 rel.
    assert chi2 == pytest.approx(2.0, rel=2e-3)
    assert chi2 < 2.0  # a symmetric dex sigma overstates the upward distance


def _chi2(sim, tru, sigma, correlation=None):
    """Chi-squared from `eval_obj`, undoing its -log10 warp (exact while chi2 >> 1e-10)."""
    whitener = None if correlation is None else CorrelationWhitener(list(tru), correlation)
    return 10 ** (-objective_mod.eval_obj(sim, tru, sigma, whitener).item())


@pytest.mark.physics_invariant
def test_eval_obj_correlation_matches_bivariate_chi_squared(monkeypatch):
    """Two correlated observables give (u1^2 - 2 rho u1 u2 + u2^2) / (1 - rho^2),
    with u = (sim - true) / sigma, whatever order they are listed in.
    """
    monkeypatch.setattr(objective_mod, 'variable_is_logarithmic', lambda key: False)
    tru = {'R_obs': 6.0e6, 'T_obs': 400.0}
    sigma = {'R_obs': 5.0e5, 'T_obs': 25.0}
    sim = {'R_obs': 7.0e6, 'T_obs': 450.0}  # u = (2, 2)
    corr = {'R_obs': {'T_obs': 0.6}}

    # (4 - 4.8 + 4) / 0.64 = 5; ignoring rho gives 8, flipping its sign 20.
    assert _chi2(sim, tru, sigma, corr) == pytest.approx(5.0, rel=1e-9)
    assert _chi2(sim, tru, sigma) == pytest.approx(8.0, rel=1e-9)
    # Residuals and pair listed in reverse order.
    sim_rev = {'T_obs': 450.0, 'R_obs': 7.0e6}
    assert _chi2(sim_rev, tru, sigma, {'T_obs': {'R_obs': 0.6}}) == pytest.approx(5.0, rel=1e-9)
    # rho = 0 reproduces the independent chi-squared.
    assert _chi2(sim, tru, sigma, {'R_obs': {'T_obs': 0.0}}) == pytest.approx(8.0, rel=1e-9)
    # A residual against the correlation, u = (2, -2), costs more: (4 + 4.8 + 4) / 0.64.
    anti = {'R_obs': 7.0e6, 'T_obs': 350.0}
    assert _chi2(anti, tru, sigma, corr) == pytest.approx(20.0, rel=1e-9)

    with pytest.raises(ValueError, match='needs sigma'):
        objective_mod.eval_obj(sim, tru, None, CorrelationWhitener(list(tru), corr))


@pytest.mark.physics_invariant
def test_eval_obj_correlation_kept_for_log_observables():
    """A log-space observable's sigma is converted to dex, its correlation used unchanged."""
    # atm_kg_per_mol is log-scaled, R_obs linear (utils.coupler).
    tru = {'atm_kg_per_mol': 0.02, 'R_obs': 6.0e6}
    sigma = {'atm_kg_per_mol': 2.0e-5, 'R_obs': 1.0e5}
    # One sigma above the target on both, so u = (1, 1) exactly.
    sim = {'atm_kg_per_mol': 0.02 * 10 ** (2.0e-5 / (0.02 * math.log(10.0))), 'R_obs': 6.1e6}

    chi2 = _chi2(sim, tru, sigma, {'atm_kg_per_mol': {'R_obs': -0.5}})

    assert chi2 == pytest.approx((1 + 1 + 1) / (1 - 0.25), rel=1e-9)  # 4
    assert abs(chi2 - 2.0) > 0.5  # rho ignored


def test_eval_obj_compares_element_ratios_in_log_only_with_sigma():
    """Without sigma C/O_atm stays linear, as before; with sigma it is compared in dex."""
    tru, sim = {'C/O_atm': 0.5}, {'C/O_atm': 0.25}

    # Linear relative residual 1 - 0.25/0.5 = 0.5; a log one would be 1 - 2 = -1.
    assert _chi2(sim, tru, None) == pytest.approx(0.25, rel=1e-6)

    # Log: (log10 0.5) / (0.05 / (0.5 ln 10)) = 6.93, so chi2 = 48.0; linear would give 25.
    expected = (math.log10(2.0) * 0.5 * math.log(10.0) / 0.05) ** 2
    assert _chi2(sim, tru, {'C/O_atm': 0.05}) == pytest.approx(expected, rel=1e-9)
    assert abs(expected - 25.0) > 20.0


def test_validate_sigma_accepts_complete_table_and_rejects_bad_entries():
    """``validate_sigma`` returns the uncertainties as floats when every
    observable has one, passes None through, and refuses a table with a
    missing, unknown, non-positive or non-finite entry, or a log-scaled
    observable whose value cannot be converted to dex.
    """
    obs = {'R_obs': 6.0e6, 'atm_kg_per_mol': 0.02}

    out = objective_mod.validate_sigma(obs, {'atm_kg_per_mol': 1, 'R_obs': 1.0e5})
    assert out == {'atm_kg_per_mol': pytest.approx(1.0), 'R_obs': pytest.approx(1.0e5)}
    assert all(isinstance(v, float) for v in out.values())
    # No table selects the relative-difference objective.
    assert objective_mod.validate_sigma(obs, None) is None

    with pytest.raises(ValueError, match='atm_kg_per_mol'):
        objective_mod.validate_sigma(obs, {'R_obs': 1.0e5})
    with pytest.raises(ValueError, match='not observables.*T_obs'):
        objective_mod.validate_sigma(obs, {'R_obs': 1.0e5, 'atm_kg_per_mol': 1e-3, 'T_obs': 5})
    for bad in (0.0, -1.0e5, float('nan'), float('inf')):
        with pytest.raises(ValueError, match="sigma for 'R_obs'"):
            objective_mod.validate_sigma(obs, {'R_obs': bad, 'atm_kg_per_mol': 1e-3})
    # A zero target on a log-scaled observable has no finite dex conversion.
    with pytest.raises(ValueError, match='log space'):
        objective_mod.validate_sigma(
            {'R_obs': 6.0e6, 'atm_kg_per_mol': 0.0}, {'R_obs': 1.0e5, 'atm_kg_per_mol': 1e-3}
        )
    # The same zero on a linear observable is fine: no conversion is made.
    lin = objective_mod.validate_sigma({'R_obs': 0.0}, {'R_obs': 1.0e5})
    assert lin['R_obs'] == pytest.approx(1.0e5)


def test_validate_sigma_takes_dex_strings_for_log_compared_observables():
    """'0.1 dex' is used as exactly 0.1 dex: a simulated C/O 0.1 dex above the
    target gives chi2 = 1 (J = 0) and 0.2 dex below gives chi2 = 4. Dex on a linearly
    compared observable, and strings that are not '<number> dex', are refused."""
    obs = {'R_obs': 6.0e6, 'C/O_atm': 0.33}
    out = objective_mod.validate_sigma(obs, {'R_obs': 1.0e5, 'C/O_atm': ' 0.1 dex '})
    assert out['C/O_atm'] == pytest.approx(0.1 * 0.33 * math.log(10.0))
    sim = {'R_obs': 6.0e6, 'C/O_atm': 0.33 * 10**0.1}
    J = objective_mod.eval_obj(sim, obs, sigma=out).item()
    assert J == pytest.approx(0.0, abs=1e-9)
    # Two sigma off in dex gives chi2 = 4, not 2 (a missing square) or 1 (no scaling)
    sim['C/O_atm'] = 0.33 * 10**-0.2
    J = objective_mod.eval_obj(sim, obs, sigma=out).item()
    assert J == pytest.approx(-math.log10(4.0), abs=1e-9)

    with pytest.raises(ValueError, match="'R_obs' is compared linearly"):
        objective_mod.validate_sigma(obs, {'R_obs': '0.1 dex', 'C/O_atm': '0.1 dex'})
    for bad in ('abc dex', '0.1 bar', '-0.1 dex', '0 dex', 'nan dex'):
        with pytest.raises(ValueError, match="sigma for 'C/O_atm'"):
            objective_mod.validate_sigma(obs, {'R_obs': 1.0e5, 'C/O_atm': bad})


def test_validate_sigma_warns_when_a_linear_sigma_converts_poorly_to_dex(caplog):
    """A linear sigma above 30 % of a log-compared value logs a warning that suggests
    dex; 20 %, a dex entry and a large sigma on a linear observable do not."""
    obs = {'R_obs': 6.0e6, 'C/O_atm': 0.5}
    logger = 'fwl.proteus.inference.objective'
    with caplog.at_level('WARNING', logger=logger):
        objective_mod.validate_sigma(obs, {'R_obs': 4.0e6, 'C/O_atm': 0.1})
        objective_mod.validate_sigma(obs, {'R_obs': 1.0e5, 'C/O_atm': '0.3 dex'})
    assert caplog.records == []
    with caplog.at_level('WARNING', logger=logger):
        objective_mod.validate_sigma(obs, {'R_obs': 1.0e5, 'C/O_atm': 0.2})
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "sigma for 'C/O_atm' is 40% of its value" in messages[0]


def test_prot_builder_unnormalizes_and_calls_J(monkeypatch):
    """``prot_builder`` returns a closure that un-normalises an x in
    [0, 1]^d to the physical parameter ranges (so x=0.5 with bounds
    [0, 10] maps to 5.0) before calling the inner objective ``J``.
    """
    captured = {}

    def fake_J(x, **kwargs):
        captured['x'] = x
        captured['kwargs'] = kwargs
        return torch.tensor([[5.0]], dtype=torch.double)

    monkeypatch.setattr(objective_mod, 'J', fake_J)

    f = objective_mod.prot_builder(
        parameters={'a': [0.0, 10.0], 'b': [2.0, 4.0]},
        observables={'obs': 1.0},
        worker=7,
        iter=9,
        output='out_dir',
        ref_config='ref.toml',
        failure_codes=[0, 1],
    )

    y = f(torch.tensor([[0.5, 0.25]], dtype=torch.double))

    assert y.item() == pytest.approx(5.0)
    assert captured['x'][0, 0].item() == pytest.approx(5.0)
    assert captured['x'][0, 1].item() == pytest.approx(2.5)


def test_prot_builder_unnormalizes_log_scaled_parameter(monkeypatch):
    """Surface pressure spans orders of magnitude, so log scaling must round-trip."""
    captured = {}

    def fake_J(x, **kwargs):
        captured['x'] = x
        return torch.tensor([[1.0]], dtype=torch.double)

    monkeypatch.setattr(objective_mod, 'J', fake_J)

    f = objective_mod.prot_builder(
        parameters={'P_surf': [1e-3, 1e3], 'struct.mass_tot': [1.0, 3.0]},
        observables={'obs': 1.0},
        worker=0,
        iter=0,
        output='out_dir',
        ref_config='ref.toml',
        failure_codes=[0, 1],
    )

    f(torch.tensor([[0.5, 0.25]], dtype=torch.double))

    assert captured['x'][0, 0].item() == pytest.approx(1.0)
    assert captured['x'][0, 1].item() == pytest.approx(1.5)


# ============================================================================
# Failure reporting for a single simulator run
# ============================================================================


def test_run_proteus_failure_distinguishes_a_missing_status_from_a_generic_error(
    monkeypatch, tmp_path
):
    """A run that dies before writing a status file is reported as having
    written none, rather than as a generic configuration error. The two call
    for different investigations: the first points at the simulator's start-up
    (environment, reference data), the second at the model configuration.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)

    def _fake_run(*_args, **kwargs):
        kwargs['stdout'].write('Error: no\n')
        kwargs['stdout'].flush()
        raise subprocess.CalledProcessError(returncode=1, cmd=['proteus'])

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)

    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=3,
            iter=4,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    failure = excinfo.value
    assert failure.status == objective_mod.STATUS_MISSING
    assert 'no readable status file' in failure.status_desc
    # Discrimination: the previous behaviour reported code 20 for this case,
    # which reads as a configuration fault the user does not have.
    assert failure.status != 20
    assert 'Generic' not in failure.status_desc
    # No logfile exists either, so the report must not invent one.
    assert failure.log_path is None
    assert 'logfile' not in failure.report()

    # Edge case: the same run with a status file present reports that status,
    # which proves the sentinel above came from the absent file and not from a
    # reader that always fails.
    (out_abs / 'status').write_text('21\nError (Interior model)\n', encoding='utf-8')
    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=3,
            iter=4,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    assert excinfo.value.status == 21


def test_run_proteus_failure_points_at_the_simulator_logfile(monkeypatch, tmp_path):
    """When the failed run left a logfile, the report names it. That file holds
    the traceback the simulator captured for itself, and is the only place the
    cause of a mid-run crash is recorded.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    (out_abs / 'proteus_00.log').write_text('early\n', encoding='utf-8')
    (out_abs / 'proteus_01.log').write_text('CRITICAL Uncaught exception\n', encoding='utf-8')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        objective_mod.subprocess,
        'run',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(returncode=1, cmd=['proteus'])
        ),
    )

    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=0,
            iter=0,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )
    # The newest logfile is the one the failed run wrote; an earlier one
    # belongs to a previous attempt in the same folder.
    assert excinfo.value.log_path.endswith('proteus_01.log')
    assert 'proteus_01.log' in excinfo.value.report()


def test_run_proteus_reports_a_clean_exit_that_produced_no_output(monkeypatch, tmp_path):
    """A run that exits zero but writes no readable helpfile is reported as a
    failed sample rather than crashing the study with a bare parser error. That
    covers a missing file, an empty one, one corrupted into invalid UTF-8 and a
    ragged row. The exit code is recorded as zero so the report does not suggest
    a crash.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    helpfile = out_abs / 'runtime_helpfile.csv'
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)
    run = functools.partial(
        objective_mod.run_proteus,
        parameters={},
        worker=0,
        iter=0,
        observables=['P_surf'],
        ref_config='reference.toml',
        output='dummy_output',
    )

    # No helpfile at all.
    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        run()
    assert excinfo.value.exit_code == 0
    assert 'no readable output' in excinfo.value.reason

    # Edge case: a helpfile that exists but holds no rows.
    helpfile.write_text('', encoding='utf-8')
    with pytest.raises(objective_mod.ProteusRunFailure):
        run()

    # Edge case: bytes that are not valid UTF-8 fail the reader's own UTF-8 read.
    helpfile.write_bytes(b'Time P_surf\n1.0 2.0\n3.0 \xff\xfe\n')
    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        run()
    assert isinstance(excinfo.value.__cause__, objective_mod.HelpfileFormatError)
    assert isinstance(excinfo.value.__cause__.__cause__, UnicodeDecodeError)

    # Edge case: a row with more fields than the header would be read one column off.
    helpfile.write_text('Time P_surf\n1.0 2.0\n3.0 4.0 5.0\n', encoding='utf-8')
    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        run()
    assert isinstance(excinfo.value.__cause__, objective_mod.HelpfileFormatError)
    assert excinfo.value.exit_code == 0
    # The reason that reaches failures.csv names the file and the line.
    assert 'runtime_helpfile.csv, line 3' in excinfo.value.reason

    # Edge case: a header with no rows is a failed sample, not an IndexError.
    helpfile.write_text('Time P_surf\n', encoding='utf-8')
    with pytest.raises(objective_mod.ProteusRunFailure) as excinfo:
        run()
    assert '0 data rows' in excinfo.value.reason

    # Discrimination: a helpfile with a usable row completes normally, so the
    # failures above come from the output and not from an unconditional raise.
    pd.DataFrame([{'P_surf': 2.5}]).to_csv(helpfile, sep=' ', index=False)
    obs, _status = run()
    assert obs['P_surf'] == pytest.approx(2.5)


def test_J_scores_a_failed_run_badly_and_keeps_the_study_running(monkeypatch, tmp_path, caplog):
    """A parameter combination the simulator cannot integrate is scored as a
    poor sample so the sweep continues, and the failure is reported once in
    full and recorded for the end-of-study tally. Aborting instead would end a
    study on the first unphysical corner of the parameter box.
    """
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )
    failure = objective_mod.ProteusRunFailure(
        reason='the simulator exited with an error',
        worker=1,
        iter=2,
        out_dir='/study/workers/w_1/i_2',
        exit_code=1,
        status=21,
        parameters={'planet.mass_tot': 3.0},
    )

    def _fail(**_kwargs):
        raise failure

    monkeypatch.setattr(objective_mod, 'run_proteus', _fail)
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '0')

    with caplog.at_level('WARNING'):
        value = objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'P_surf': 1.0},
            worker=1,
            iter=2,
            output='dummy_output',
            ref_config='reference.toml',
        )

    assert value.shape == (1, 1)
    assert value.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # The score must be far below any value a successful run can produce, or
    # the optimiser would be drawn toward the region that fails.
    assert value.item() < -10.0
    # Reported once, in full: the status description and the output folder are
    # what let the user find the run.
    reported = '\n'.join(record.getMessage() for record in caplog.records)
    assert 'Interior model' in reported
    assert '/study/workers/w_1/i_2' in reported

    # The same failure is left on disk for the end-of-study tally, because a
    # log line scrolls past and a study that failed mostly needs a count.
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [(r['worker'], r['iter'], r['status']) for r in recorded] == [(1, 2, 21)]
    assert recorded[0]['planet.mass_tot'] == pytest.approx(3.0)

    # Opting in turns the same failure into a hard stop; monkeypatch restores
    # the variable afterwards.
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '1')
    with pytest.raises(objective_mod.ProteusRunFailure):
        objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'P_surf': 1.0},
            worker=1,
            iter=2,
            output='dummy_output',
            ref_config='reference.toml',
        )


def test_J_scores_a_clean_run_that_stopped_in_an_error_state(monkeypatch, tmp_path, caplog):
    """A run that exits cleanly but records an error status is scored badly and
    named in the log. Status 25 is the only error code reachable this way: it
    is written when a run is stopped through its keepalive file, and the
    simulator then terminates normally.

    'R_obs' is used as the observable because it is compared linearly, which
    gives the exact-match objective a closed form to pin against.
    """
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )
    monkeypatch.setattr(
        objective_mod,
        'run_proteus',
        lambda **_kwargs: ({'R_obs': 9.25e6}, 25),
    )
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '0')

    with caplog.at_level('WARNING'):
        value = objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=0,
            iter=0,
            output='dummy_output',
            ref_config='reference.toml',
        )
    assert value.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert 'status 25' in '\n'.join(r.getMessage() for r in caplog.records)
    # Counted in the end-of-study tally alongside the runs that crashed. A
    # tally that covered only crashes would understate a study stopped by hand.
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [r['status'] for r in recorded] == [25]
    assert recorded[0]['exit_code'] == 0

    # Discrimination: the same observables under a completed status (13,
    # "target time reached") are scored normally, which rules out a regression
    # that returns the failure score for every run.
    monkeypatch.setattr(
        objective_mod,
        'run_proteus',
        lambda **_kwargs: ({'R_obs': 9.25e6}, 13),
    )
    good = objective_mod.J(
        x=torch.tensor([[0.5]], dtype=torch.double),
        parameters=['planet.mass_tot'],
        true_observables={'R_obs': 9.25e6},
        worker=0,
        iter=0,
        output='dummy_output',
        ref_config='reference.toml',
        failure_codes=[],
    )
    # Closed form for an exact match on a linear observable: the normalised
    # difference is zero, so sq_dist is zero and the score is
    # -log10(0 + EPS_CLIP) = -log10(1e-10) = 10.
    assert good.item() == pytest.approx(10.0, rel=1e-9)
    # Sign guard: a flipped objective would land at -10, which is still above
    # BAD_OBJ_VALUE and would pass a bare "better than failure" assertion.
    assert good.item() > 0
    # Scale guard: the failure score is -20, so the two are far apart.
    assert good.item() - objective_mod.BAD_OBJ_VALUE > 25.0


def test_J_aborts_on_a_clean_run_that_stopped_in_an_error_state(monkeypatch, tmp_path):
    """`abort_on_failure` stops the study on a run that exited cleanly but
    recorded an error status, the same way it stops on a run that crashed.
    Both are faults; only the route by which the simulator reported them
    differs, so honouring the setting on one and not the other would let a
    study set up with `abort_on_failure = true` run to completion on a
    reference config that fails every evaluation.

    The asymmetry the setting must keep: an excluded status completed
    normally, so it is scored as a poor sample and the study carries on even
    with aborting enabled.
    """
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )

    def _run(status, worker, iter, codes=()):
        monkeypatch.setattr(
            objective_mod,
            'run_proteus',
            lambda **_kwargs: ({'R_obs': 9.25e6}, status),
        )
        return objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=worker,
            iter=iter,
            output='dummy_output',
            ref_config='reference.toml',
            failure_codes=list(codes),
        )

    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '1')

    # Status 25: written when a run is stopped through its keepalive file, so
    # the simulator exits 0 and the fault is visible only in the status file.
    with pytest.raises(objective_mod.ProteusRunFailure) as caught:
        _run(25, worker=0, iter=0)
    assert caught.value.status == 25
    assert caught.value.category == objective_mod.CATEGORY_FAILURE
    # Exit code 0 is the whole point of this path: the abort must not depend
    # on the child having exited non-zero.
    assert caught.value.exit_code == 0

    # Boundary of the failure set: STATUS_MISSING is a fault because the run's
    # own account of itself is absent. A check on `20 <= status <= 28` alone
    # would miss it.
    with pytest.raises(objective_mod.ProteusRunFailure) as missing:
        _run(objective_mod.STATUS_MISSING, worker=0, iter=1)
    assert missing.value.status == objective_mod.STATUS_MISSING

    # The record is written before the abort, so an aborted study still says
    # on disk what stopped it rather than leaving only the traceback.
    recorded = failures_mod.read_failure_records(tmp_path)
    # Ordered by (worker, iter), so the status-25 run at iter 0 comes first.
    assert [r['status'] for r in recorded] == [25, objective_mod.STATUS_MISSING]

    # Discrimination against a fix that aborts on `failed or excluded`: an
    # excluded status is scored as a poor sample and returns normally.
    excluded = _run(11, worker=1, iter=0, codes=(11,))
    assert excluded.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # Boundedness: the failure score sits far below anything a completed run
    # can reach, so the optimiser is not drawn toward the excluded region.
    assert excluded.item() < -10.0
    assert failures_mod.read_failure_records(tmp_path)[-1]['category'] == (
        objective_mod.CATEGORY_EXCLUDED
    )

    # Discrimination against a regression that raises unconditionally: with
    # the setting off, the same error status is scored and the study goes on.
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '0')
    scored = _run(25, worker=2, iter=0)
    assert scored.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert scored.item() < -10.0


def test_J_treats_the_documented_error_codes_as_failures(monkeypatch, tmp_path):
    """The failure range covers the error statuses the simulator can record."""
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '0')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )

    def _score(status, worker=0):
        monkeypatch.setattr(
            objective_mod,
            'run_proteus',
            lambda **_kwargs: ({'R_obs': 9.25e6}, status),
        )
        return objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=worker,
            iter=0,
            output='dummy_output',
            ref_config='reference.toml',
            failure_codes=[],
        ).item()

    # Highest defined error code, and the escape-model error below it.
    assert _score(28) == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert _score(21) == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # An evaporated planet is kept out of the fit with no `failure_codes` entry.
    assert _score(29, worker=1) == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    recorded = {r['worker']: r for r in failures_mod.read_failure_records(tmp_path)}
    assert (recorded[1]['status'], recorded[1]['category']) == (
        29,
        objective_mod.CATEGORY_EXCLUDED,
    )
    assert recorded[0]['category'] == objective_mod.CATEGORY_FAILURE
    # Discrimination: other completion codes are still scored on their
    # observables, -log10(0 + 1e-10) = 10 for an exact match, so status 29 is
    # not excluded by a range that also catches 13.
    assert _score(13) == pytest.approx(10.0, rel=1e-9)
    assert _score(18) == pytest.approx(10.0, rel=1e-9)
    # A run that never updated its status past 'Running' died mid-flight.
    assert _score(1) == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # An unreadable status file is treated as a failure, because the run's own
    # account of itself is missing and its output cannot be trusted.
    assert _score(objective_mod.STATUS_MISSING) == pytest.approx(objective_mod.BAD_OBJ_VALUE)


def test_J_separates_an_excluded_outcome_from_a_failed_run(monkeypatch, tmp_path, caplog):
    """A status named in `failure_codes` marks an outcome the study does not fit
    against, not a fault. A run stopped by its clock limit (status 11) completed
    normally, so it is scored as a poor sample and reported at info level, while
    an error status (21, interior model) is reported as a run that produced
    nothing usable. Reporting the first as the second sends the user looking for
    a bug in a run that did exactly what it was configured to do.
    """
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '0')
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(tmp_path)}
    )

    def _score(status, worker):
        monkeypatch.setattr(
            objective_mod,
            'run_proteus',
            lambda **_kwargs: ({'R_obs': 9.25e6}, status),
        )
        return objective_mod.J(
            x=torch.tensor([[0.5]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=worker,
            iter=0,
            output='dummy_output',
            ref_config='reference.toml',
            failure_codes=[11],
        ).item()

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.objective'):
        excluded = _score(11, worker=0)

    # The optimiser must still be steered away from the excluded region, so the
    # score is the same one a failure carries.
    assert excluded == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    assert not [r for r in caplog.records if r.levelname in ('WARNING', 'ERROR')]
    reported = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'excludes' in reported
    assert 'maximum clock runtime' in reported
    assert 'failure state' not in reported
    assert 'failed for worker=0' not in reported

    # The record is kept for the end-of-study tally, labelled so the tally can
    # count it apart from the runs that genuinely failed.
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [(r['status'], r['category']) for r in recorded] == [
        (11, objective_mod.CATEGORY_EXCLUDED)
    ]

    # Discrimination: an error status under the same call is still a failure,
    # warned about and recorded under the other category. Without this the test
    # would pass against a regression that labelled every run 'excluded'.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.objective'):
        failed = _score(21, worker=1)
    assert failed == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    assert 'failed for worker=1' in warnings[0].getMessage()
    assert 'stopped in a failure state' in warnings[0].getMessage()
    assert 'status 21' in warnings[0].getMessage()
    recorded = failures_mod.read_failure_records(tmp_path)
    assert [(r['status'], r['category']) for r in recorded] == [
        (11, objective_mod.CATEGORY_EXCLUDED),
        (21, objective_mod.CATEGORY_FAILURE),
    ]

    # Discrimination: a completion status that the study does not exclude is
    # scored on its observables and leaves no record at all. The exact match on
    # a linear observable has the closed form -log10(0 + 1e-10) = 10.
    assert _score(13, worker=2) == pytest.approx(10.0, rel=1e-9)
    assert len(failures_mod.read_failure_records(tmp_path)) == 2


# ============================================================================
# Failure records: written per evaluation, read back for the study summary
# ============================================================================


def test_run_output_dir_names_the_folder_the_simulator_is_given(monkeypatch, tmp_path):
    """The per-evaluation folder is derived in one place, so the path a failure
    report names is the path the simulator was told to write to. Initial
    samples use worker -1, which must survive the same construction.
    """
    monkeypatch.setattr(
        objective_mod,
        'get_proteus_directories',
        lambda path: {'output': str(tmp_path / path)},
    )

    rel, absolute = objective_mod.run_output_dir('study', 2, 7)
    assert rel.as_posix() == 'study/workers/w_2/i_7'
    assert absolute == tmp_path / 'study' / 'workers' / 'w_2' / 'i_7'

    # Initial sampling identifies itself with worker -1 rather than a worker
    # index, and must land in its own folder rather than colliding with w_1.
    rel_init, _ = objective_mod.run_output_dir('study', -1, 7)
    assert rel_init.as_posix() == 'study/workers/w_-1/i_7'
    assert rel_init != rel


def test_J_records_clean_exit_failures_through_the_real_simulator_wrapper(
    monkeypatch, tmp_path
):
    """Only the simulator call is replaced, so `J` drives the real `run_proteus`,
    config writing, status reading and failure table. Four evaluations cover
    the outcomes a study meets: a crash, a clean exit on an error status (25),
    a clean exit on a status the study excludes (11), and a completed run (13).

    The two clean-exit outcomes build their report from the swept values held
    by `J`. Those must stay numeric and limited to the swept keys, or the report
    cannot be formatted and the table gains columns partway through, after
    which it no longer reads back and the summary counts every run as usable.
    """
    monkeypatch.setenv('PROTEUS_OUTPUT_PATH', str(tmp_path))
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '0')
    ref_config = tmp_path / 'reference.toml'
    ref_config.write_text('[planet]\nmass_tot = 1.0\n\n[params.out]\npath = "unset"\n')

    # Status each worker's run records, and whether it then crashes.
    outcomes = {0: (21, True), 1: (25, False), 2: (11, False), 3: (13, False)}
    calls = []

    def _fake_run(command, **_kwargs):
        cfg = toml.load(command[3])
        calls.append(cfg)
        out_abs = tmp_path / cfg['params']['out']['path']
        worker = int(out_abs.parent.name.removeprefix('w_'))
        status, crashes = outcomes[worker]
        (out_abs / 'status').write_text(f'{status}\n')
        pd.DataFrame([{'P_surf': 1e5, 'R_obs': 9.25e6}]).to_csv(
            out_abs / 'runtime_helpfile.csv', sep=' ', index=False
        )
        if crashes:
            raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(objective_mod.subprocess, 'run', _fake_run)

    # A distinct swept value per worker, so each row can be matched to its run.
    scores = {
        w: objective_mod.J(
            x=torch.tensor([[1.5 + w]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=w,
            iter=0,
            output='study',
            ref_config=str(ref_config),
            failure_codes=[11],
        ).item()
        for w in outcomes
    }

    assert [scores[w] for w in (0, 1, 2)] == pytest.approx([objective_mod.BAD_OBJ_VALUE] * 3)
    # Discrimination: the completed run goes through the same wrapper and is
    # scored on its observables, -log10(0 + 1e-10) = 10 for an exact match.
    assert scores[3] == pytest.approx(10.0, rel=1e-9)

    # The fixed entries still reach the simulator config; keeping them out of
    # the report must not keep them out of the run.
    assert len(calls) == 4
    assert calls[1]['params']['out']['path'] == 'study/workers/w_1/i_0'
    assert calls[1]['params']['out']['plot_mod'] == 'none'
    assert calls[1]['atmos_clim']['spectral_cache'].endswith('spectral_cache')
    assert calls[1]['planet']['mass_tot'] == pytest.approx(2.5)

    # One header and one row per unscored run, all the fixed columns plus the
    # swept parameter. Parsed as CSV, since the status 25 description holds a
    # quoted comma.
    with open(tmp_path / 'study' / failures_mod.FAILURE_CSV, newline='') as f:
        table = list(csv.reader(f))
    header = table[0]
    assert header == [*failures_mod._FAILURE_COLUMNS, 'planet.mass_tot']
    assert 'params.out.path' not in header
    assert len(table) == 4
    assert {len(row) for row in table} == {len(header)}

    records = failures_mod.read_failure_records(tmp_path / 'study')
    assert [(r['worker'], r['status'], r['category']) for r in records] == [
        (0, 21, objective_mod.CATEGORY_FAILURE),
        (1, 25, objective_mod.CATEGORY_FAILURE),
        (2, 11, objective_mod.CATEGORY_EXCLUDED),
    ]
    assert [r['planet.mass_tot'] for r in records] == pytest.approx([1.5, 2.5, 3.5])
    assert [r['exit_code'] for r in records] == [1, 0, 0]

    # The study tally counts the three unscored runs. An unreadable table would
    # report zero here and describe every evaluation as usable.
    assert failures_mod.summarise_failures(str(tmp_path / 'study'), n_attempted=4) == 3


@pytest.fixture
def clean_dispatch(monkeypatch):
    """Keep a test's dispatch setting out of the rest of the session.

    `monkeypatch.delenv` records no undo for a variable that was unset, so the
    value is pinned with `setenv` before a test is allowed to change it.
    """
    monkeypatch.setenv('PROTEUS_INFERENCE_DISPATCH', objective_mod.DISPATCH_SUBPROCESS)
    monkeypatch.setenv('PROTEUS_INFERENCE_RUNNER_MAX_JOBS', '0')
    return monkeypatch


def _stub_runner(monkeypatch, out_abs, outcome, seen=None):
    """Point `run_proteus` at a runner returning `outcome`, recording its calls."""

    class _Runner:
        def run(self, config_path, timeout):
            if seen is not None:
                seen.append((str(config_path), timeout))
            return outcome

    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *args, **kwargs: None)
    monkeypatch.setattr(objective_mod, 'worker_runner', lambda _console: _Runner())
    monkeypatch.setenv('PROTEUS_INFERENCE_DISPATCH', 'runner')


def test_set_dispatch_rejects_an_unknown_mode(clean_dispatch):
    """Error contract: a misspelled dispatch is refused where the study is
    configured, and leaves the mode already in force untouched.
    """
    objective_mod.set_dispatch(objective_mod.DISPATCH_RUNNER, max_jobs=25)
    assert objective_mod.dispatch_mode() == 'runner'

    with pytest.raises(ValueError, match='sub-process'):
        objective_mod.set_dispatch('sub-process')

    assert objective_mod.dispatch_mode() == 'runner'


def test_dispatch_defaults_to_one_process_per_evaluation(clean_dispatch):
    """A study that asks for nothing, and one whose environment carries a value
    this version does not know, both get the long-standing behaviour.
    """
    clean_dispatch.delenv('PROTEUS_INFERENCE_DISPATCH')
    assert objective_mod.dispatch_mode() == 'subprocess'

    clean_dispatch.setenv('PROTEUS_INFERENCE_DISPATCH', 'from-a-newer-version')
    assert objective_mod.dispatch_mode() == 'subprocess'

    # None is the same as not asking, and clears a mode set earlier.
    objective_mod.set_dispatch(objective_mod.DISPATCH_RUNNER)
    objective_mod.set_dispatch(None)
    assert objective_mod.dispatch_mode() == 'subprocess'


def test_run_proteus_reuses_a_process_when_dispatch_is_runner(clean_dispatch, tmp_path):
    """The evaluation goes to the worker's reused process, no new `proteus
    start` is launched, and the observables come back as on the default path.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    pd.DataFrame([{'P_surf': 1.2e5, 'T_surf': 1400.0}]).to_csv(
        out_abs / 'runtime_helpfile.csv', sep=' ', index=False
    )

    calls = []
    _stub_runner(clean_dispatch, out_abs, outcome=None, seen=calls)
    clean_dispatch.setattr(
        objective_mod.subprocess,
        'run',
        lambda *args, **kwargs: pytest.fail('the runner dispatch must not start a new process'),
    )
    clean_dispatch.setenv('PROTEUS_INFERENCE_CHILD_TIMEOUT_S', '900')

    obs, _status = objective_mod.run_proteus(
        parameters={},
        worker=3,
        iter=4,
        observables=['P_surf', 'T_surf'],
        ref_config='reference.toml',
        output='dummy_output',
    )

    # The per-run timeout is carried through, so a wedged simulation is still
    # bounded when it runs inside a reused process.
    assert calls == [(str(out_abs / 'input.toml'), pytest.approx(900.0))]
    assert obs['P_surf'] == pytest.approx(1.2e5)
    assert obs['T_surf'] == pytest.approx(1400.0)


def test_runner_dispatch_reports_failures_like_a_failed_child(clean_dispatch, tmp_path):
    """A run the reused process could not complete carries the same failure
    type, reason and exit code as one that ran on its own, and names the
    per-worker console file the runner dispatch actually writes.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    _stub_runner(clean_dispatch, out_abs, outcome=('exceeded the 900.0 s timeout', None))

    with pytest.raises(failures_mod.ProteusRunFailure) as excinfo:
        objective_mod.run_proteus(
            parameters={},
            worker=3,
            iter=4,
            observables=['P_surf'],
            ref_config='reference.toml',
            output='dummy_output',
        )

    failure = excinfo.value
    assert failure.reason == 'exceeded the 900.0 s timeout'
    # None, not 0: stopped rather than ended on its own, and the report has to
    # keep those apart.
    assert failure.exit_code is None
    assert (failure.worker, failure.iter) == (3, 4)
    # One console file per worker, so its name carries no iteration.
    assert failure.console_path.endswith(f'runner{failures_mod.CHILD_CONSOLE_SUFFIX}')


def test_runner_dispatch_gives_each_initial_sampling_process_its_own_console(
    clean_dispatch, tmp_path
):
    """Initial samples all run as worker -1 from several pool processes, so the
    console file the runner is handed names the process. A shared name would
    interleave their output and send every failure report to one mixed file.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    _stub_runner(clean_dispatch, out_abs, outcome=('exceeded the 900.0 s timeout', None))
    consoles = []
    runner = objective_mod.worker_runner
    clean_dispatch.setattr(
        objective_mod,
        'worker_runner',
        lambda console: consoles.append(console) or runner(console),
    )

    def failed_console(pid):
        clean_dispatch.setattr(objective_mod.os, 'getpid', lambda: pid)
        with pytest.raises(failures_mod.ProteusRunFailure) as excinfo:
            objective_mod.run_proteus(
                parameters={},
                worker=-1,
                iter=0,
                observables=['P_surf'],
                ref_config='reference.toml',
                output='dummy_output',
            )
        return excinfo.value.console_path

    first, second = failed_console(1111), failed_console(2222)

    assert first != second
    assert first.endswith(f'runner_1111{failures_mod.CHILD_CONSOLE_SUFFIX}')
    assert second.endswith(f'runner_2222{failures_mod.CHILD_CONSOLE_SUFFIX}')
    # The report points at the file the runner was given, not a guess at it.
    assert [str(c) for c in consoles] == [first, second]


def test_run_proteus_leaves_the_callers_parameter_dict_as_it_was_passed(monkeypatch, tmp_path):
    """The run-specific config entries the simulator needs are added to the
    config it is given, not to the dict the caller passed in. The caller keeps
    that dict to describe the point it evaluated, so a string-valued config
    entry appearing in it turns the record of a swept point into a mixture of
    swept values and fixed plumbing.
    """
    out_abs = tmp_path / 'sim'
    out_abs.mkdir(parents=True)
    pd.DataFrame([{'P_surf': 1.0e5, 'R_obs': 9.25e6}]).to_csv(
        out_abs / 'runtime_helpfile.csv', sep=' ', index=False
    )

    written = []
    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(out_abs)}
    )
    monkeypatch.setattr(
        objective_mod,
        'update_toml',
        lambda _config_file, values, _output_file: written.append(dict(values)),
    )
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)

    swept = {'planet.mass_tot': 3.25, 'struct.corefrac': 0.55}
    before = dict(swept)
    objective_mod.run_proteus(
        parameters=swept,
        worker=1,
        iter=2,
        observables=['R_obs'],
        ref_config='reference.toml',
        output='dummy_output',
    )

    assert swept == before
    # Named explicitly, because these are the entries that carry strings and so
    # the ones whose leakage breaks a numeric-only consumer.
    assert 'params.out.path' not in swept
    assert 'params.out.plot_mod' not in swept

    # Discrimination: the entries are absent from the caller's dict because
    # they went into the copy, not because they stopped being written. A
    # regression that dropped the injection would leave the simulator writing
    # into the reference config's own output folder.
    config = written[0]
    assert config['params.out.path'].endswith('workers/w_1/i_2')
    assert config['params.out.plot_mod'] == 'none'
    assert config['planet.mass_tot'] == pytest.approx(3.25)

    # Edge case: a study that sweeps nothing still gets a fully populated
    # config, and still gets its empty dict back unchanged.
    empty = {}
    objective_mod.run_proteus(
        parameters=empty,
        worker=1,
        iter=2,
        observables=['R_obs'],
        ref_config='reference.toml',
        output='dummy_output',
    )
    assert empty == {}
    assert set(objective_mod._FIXED_PARAMETER_KEYS) <= set(written[-1])


@pytest.mark.physics_invariant
def test_J_reports_a_clean_exit_failure_with_only_the_swept_values(
    monkeypatch, tmp_path, caplog
):
    """A run that exits cleanly on an error status is scored, recorded and
    reported in full, with the swept point named and the fixed config entries
    left out. The report is built on every such evaluation whatever the log
    level, so a value it cannot render ends the worker that was evaluating it.

    The simulator is driven through the real wrapper rather than a stand-in, so
    the parameter dict the report receives is the one the wrapper leaves behind.
    """
    study = tmp_path / 'study'
    study.mkdir(parents=True)
    pd.DataFrame([{'P_surf': 1.0e5, 'R_obs': 9.25e6}]).to_csv(
        study / 'runtime_helpfile.csv', sep=' ', index=False
    )
    # Status 25: stopped through the keepalive file, which the simulator
    # reports by writing the code and then terminating normally.
    (study / 'status').write_text('25\n')

    monkeypatch.setattr(
        objective_mod, 'get_proteus_directories', lambda _path: {'output': str(study)}
    )
    monkeypatch.setattr(objective_mod, 'update_toml', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(objective_mod.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setenv(failures_mod.ABORT_ON_FAILURE_ENV, '0')

    with caplog.at_level(logging.DEBUG, logger='fwl.proteus.inference.objective'):
        value = objective_mod.J(
            x=torch.tensor([[3.25]], dtype=torch.double),
            parameters=['planet.mass_tot'],
            true_observables={'R_obs': 9.25e6},
            worker=1,
            iter=2,
            output='dummy_output',
            ref_config='reference.toml',
        )

    assert value.item() == pytest.approx(objective_mod.BAD_OBJ_VALUE)
    # Sign and scale guards: the failure score sits below anything the warped
    # objective can return for a real match, whose ceiling is -log10(EPS_CLIP).
    assert value.item() < 0
    assert (
        objective_mod.BAD_OBJ_VALUE
        < -objective_mod.log_warp(torch.zeros((1, 1), dtype=torch.double)).item()
    )

    reported = '\n'.join(record.getMessage() for record in caplog.records)
    assert 'planet.mass_tot=3.25' in reported
    # The fixed entries are plumbing, identical for every evaluation, and one
    # of them is the study-wide spectral cache path. A report naming them tells
    # the reader nothing and hides the point that actually failed.
    assert 'spectral_cache' not in reported
    assert 'plot_mod' not in reported

    recorded = failures_mod.read_failure_records(study)
    assert [r['status'] for r in recorded] == [25]
    assert recorded[0]['planet.mass_tot'] == pytest.approx(3.25)
    assert 'params.out.path' not in recorded[0]
