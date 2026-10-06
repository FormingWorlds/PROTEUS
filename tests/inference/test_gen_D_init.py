"""
Unit tests for inference initial-dataset generation utilities.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
import toml

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
torch = pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

import proteus.inference.gen_D_init as init_mod  # noqa: E402
from proteus.inference.failures import ProteusRunFailure  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_create_init_falls_back_to_n_workers_when_init_samps_less_than_one(monkeypatch):
    """When ``init_samps < 1``, ``create_init`` falls back to the number of
    workers rather than raising an error. The fallback allows the caller to
    omit an explicit sample count when the worker count is the natural default.
    """
    received: list = []

    def _mock_bounds(
        output,
        ref_config,
        parameters,
        observables,
        n,
        seed,
        n_workers,
        failure_codes,
        sigma=None,
        correlation=None,
    ):
        received.append(n)
        return n

    monkeypatch.setattr(init_mod, 'sample_from_bounds', _mock_bounds)

    config = {
        'init_grid': 'none',
        'init_samps': 0,
        'output': 'out',
        'ref_config': 'ref.toml',
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
        'observables': {'R_obs': 1.0},
        'seed': 1,
        'n_workers': 4,
        'failure_codes': [],
    }
    result = init_mod.create_init(config)

    assert received == [4], 'expected sample_from_bounds to be called with n_workers=4'
    assert result == 4


@pytest.mark.unit
def test_create_init_routes_to_sample_from_bounds(monkeypatch):
    """``create_init`` with ``init_grid='none'`` dispatches to
    ``sample_from_bounds`` (Halton-sequence sampling of the parameter
    box), not to ``sample_from_grid``.
    """
    config = {
        'init_grid': 'none',
        'init_samps': 4,
        'output': 'out',
        'ref_config': 'ref.toml',
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
        'observables': {'R_obs': 1.0},
        'seed': 1,
        'n_workers': 2,
        'failure_codes': [],
    }
    grid_calls: list = []
    monkeypatch.setattr(
        init_mod, 'sample_from_grid', lambda *a, **kw: grid_calls.append((a, kw))
    )
    monkeypatch.setattr(init_mod, 'sample_from_bounds', lambda *args, **kwargs: 4)
    assert init_mod.create_init(config) == 4
    # Discrimination: with init_grid='none' the wrapper must dispatch ONLY to
    # sample_from_bounds. A regression that called both backends would still
    # return 4 from the bounds shim.
    assert grid_calls == []


@pytest.mark.unit
def test_create_init_routes_to_sample_from_grid(monkeypatch, tmp_path):
    """``create_init`` with a non-'none' ``init_grid`` dispatches to
    ``sample_from_grid`` and resolves the grid path through the output root,
    so a relocated output root (PROTEUS_OUTPUT_PATH) is honoured rather than
    hard-coding ``<proteus>/output``.
    """
    observed = {}
    # Model the output root as <tmp_path>/output; the grid name is appended to
    # it, mirroring get_proteus_directories(outdir)['output'].
    monkeypatch.setattr(
        init_mod,
        'get_proteus_directories',
        lambda outdir: {'output': str(tmp_path / 'output' / outdir)},
    )

    def fake_sample_from_grid(
        output, params, observables, grid_dir, sigma=None, correlation=None
    ):
        observed['grid_dir'] = grid_dir
        return 6

    monkeypatch.setattr(init_mod, 'sample_from_grid', fake_sample_from_grid)
    config = {
        'init_grid': 'my_grid',
        'init_samps': 3,
        'output': 'out',
        'parameters': {'planet.mass_tot': [0.7, 3.0]},
        'observables': {'R_obs': 1.0},
        'failure_codes': [],
    }

    assert init_mod.create_init(config) == 6
    # The grid dir is the (possibly relocated) output root joined with the grid
    # name. The full path rules out ignoring the grid name or a hard-coded
    # <proteus>/output root.
    assert observed['grid_dir'] == str(tmp_path / 'output' / 'my_grid')


@pytest.mark.unit
def test_sample_from_grid_builds_and_saves_dataset(monkeypatch, tmp_path):
    """``sample_from_grid`` walks each ``case_N/`` subdirectory, reads
    the case parameters from ``init_coupler.toml``, reads the observable
    from ``runtime_helpfile.csv``, and saves the combined dataset as
    ``init.csv`` with canonical columns ``x_0, y``.
    """
    grid_dir = tmp_path / 'grid'
    output_dir = tmp_path / 'output'
    output_dir.mkdir(parents=True)
    for i, mass in enumerate([1.0, 2.0]):
        case = grid_dir / f'case_{i}'
        case.mkdir(parents=True)
        pd.DataFrame([{'R_obs': 1.5 + i}]).to_csv(
            case / 'runtime_helpfile.csv', sep=' ', index=False
        )
        (case / 'init_coupler.toml').write_text(
            toml.dumps({'planet': {'mass_tot': mass}}), encoding='utf-8'
        )

    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(output_dir)}
    )

    n = init_mod.sample_from_grid(
        output='ignored',
        params={'planet.mass_tot': [0.0, 10.0]},
        observables={'R_obs': 1.0},
        grid_dir=str(grid_dir),
    )

    data = pd.read_csv(output_dir / 'init.csv')
    assert n == 2
    assert list(data.columns) == ['x_0', 'y']
    assert len(data) == 2


@pytest.mark.unit
def test_sample_from_grid_skips_a_case_whose_helpfile_row_is_ragged(
    monkeypatch, tmp_path, caplog
):
    """A case whose helpfile row has more fields than its header is skipped with a
    warning, so one unreadable case does not stop the dataset from the others."""
    grid_dir = tmp_path / 'grid'
    output_dir = tmp_path / 'output'
    output_dir.mkdir(parents=True)
    rows = ['R_obs\n1.5\n4.5 9.0\n', 'R_obs\n2.5\n', 'R_obs\n3.5\n']
    for i, (mass, text) in enumerate(zip([1.0, 2.0, 3.0], rows)):
        case = grid_dir / f'case_{i}'
        case.mkdir(parents=True)
        (case / 'runtime_helpfile.csv').write_text(text, encoding='utf-8')
        (case / 'init_coupler.toml').write_text(
            toml.dumps({'planet': {'mass_tot': mass}}), encoding='utf-8'
        )
    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(output_dir)}
    )

    with caplog.at_level('WARNING'):
        n = init_mod.sample_from_grid(
            output='ignored',
            params={'planet.mass_tot': [0.0, 10.0]},
            observables={'R_obs': 1.0},
            grid_dir=str(grid_dir),
        )

    assert n == 2
    assert 'Skipping case_0' in caplog.text
    # The surviving cases keep their own configs: masses 2 and 3, not 1 and 2.
    assert pd.read_csv(output_dir / 'init.csv')['x_0'].tolist() == pytest.approx([0.2, 0.3])


@pytest.mark.unit
def test_sample_from_grid_skips_a_case_without_a_helpfile(monkeypatch, tmp_path, caplog):
    """A case with no helpfile, or with a directory in its place, is skipped like an
    unreadable one. The directory case pins the OSError half of the except clause."""
    grid_dir = tmp_path / 'grid'
    for i, text in enumerate([None, 'dir', 'R_obs\n2.5\n']):
        case = grid_dir / f'case_{i}'
        case.mkdir(parents=True)
        if text == 'dir':
            (case / 'runtime_helpfile.csv').mkdir()
        elif text is not None:
            (case / 'runtime_helpfile.csv').write_text(text, encoding='utf-8')
        (case / 'init_coupler.toml').write_text(f'[planet]\nmass_tot = {i + 2.0}\n')
    output_dir = tmp_path / 'out'
    output_dir.mkdir()
    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(output_dir)}
    )

    with caplog.at_level('WARNING'):
        n = init_mod.sample_from_grid(
            output='ignored',
            params={'planet.mass_tot': [0.0, 10.0]},
            observables={'R_obs': 1.0},
            grid_dir=str(grid_dir),
        )

    assert n == 1
    assert 'Skipping case_0' in caplog.text and 'Skipping case_1' in caplog.text
    assert pd.read_csv(output_dir / 'init.csv')['x_0'].tolist() == pytest.approx([0.4])


@pytest.mark.unit
def test_sample_from_grid_refuses_a_grid_with_no_readable_case(monkeypatch, tmp_path, caplog):
    """A grid whose cases are all empty or header-only raises instead of writing an
    empty dataset, and the warning for each case names it."""
    grid_dir = tmp_path / 'grid'
    for i, text in enumerate(['', 'R_obs\n']):
        case = grid_dir / f'case_{i}'
        case.mkdir(parents=True)
        (case / 'runtime_helpfile.csv').write_text(text, encoding='utf-8')
        (case / 'init_coupler.toml').write_text('[planet]\nmass_tot = 1.0\n')
    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(tmp_path / 'out')}
    )

    with caplog.at_level('WARNING'), pytest.raises(ValueError, match='No readable helpfile'):
        init_mod.sample_from_grid(
            output='ignored',
            params={'planet.mass_tot': [0.0, 10.0]},
            observables={'R_obs': 1.0},
            grid_dir=str(grid_dir),
        )

    assert 'Skipping case_0' in caplog.text and 'Skipping case_1' in caplog.text
    assert not (tmp_path / 'out' / 'init.csv').exists()


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_sample_from_grid_scores_with_observable_correlation(monkeypatch, tmp_path):
    """Grid cases are scored by the correlated chi-squared, with the helpfile row
    (a pandas Series) matched to the matrix by name: u = (2, 2), rho = 0.6 gives 5.
    """
    grid_dir = tmp_path / 'grid'
    output_dir = tmp_path / 'output'
    output_dir.mkdir(parents=True)
    case = grid_dir / 'case_0'
    case.mkdir(parents=True)
    pd.DataFrame([{'T_obs': 450.0, 'R_obs': 7.0e6}]).to_csv(
        case / 'runtime_helpfile.csv', sep=' ', index=False
    )
    (case / 'init_coupler.toml').write_text(
        toml.dumps({'planet': {'mass_tot': 1.0}}), encoding='utf-8'
    )
    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(output_dir)}
    )

    init_mod.sample_from_grid(
        output='ignored',
        params={'planet.mass_tot': [0.0, 10.0]},
        observables={'R_obs': 6.0e6, 'T_obs': 400.0},
        grid_dir=str(grid_dir),
        sigma={'R_obs': 5.0e5, 'T_obs': 25.0},
        correlation={'R_obs': {'T_obs': 0.6}},
    )

    y = pd.read_csv(output_dir / 'init.csv')['y'].iloc[0]
    assert y == pytest.approx(-np.log10(5.0 + 1e-10), rel=1e-9)
    # Correlation guard: the independent chi-squared of 8 is 0.2 lower.
    assert abs(y + np.log10(8.0)) > 0.1


@pytest.mark.unit
def test_sample_from_bounds_rejects_invalid_worker_count():
    """``sample_from_bounds`` rejects ``n_workers < 1`` with an
    'at least 1' message, so a misconfigured worker pool fails loudly
    rather than silently producing zero samples.
    """
    with pytest.raises(ValueError, match='at least 1'):
        init_mod.sample_from_bounds(
            output='out',
            ref_config='ref.toml',
            params={'a': [0.0, 1.0]},
            observables={'obs': 1.0},
            nsamp=2,
            seed=1,
            n_workers=0,
            failure_codes=[],
        )
    # Discrimination: negative counts raise too. A guard on n_workers == 0
    # alone would let -1 reach multiprocessing.Pool and fail opaquely there.
    with pytest.raises(ValueError, match='at least 1'):
        init_mod.sample_from_bounds(
            output='out',
            ref_config='ref.toml',
            params={'a': [0.0, 1.0]},
            observables={'obs': 1.0},
            nsamp=2,
            seed=1,
            n_workers=-1,
            failure_codes=[],
        )
    # Discrimination: negative counts raise too. A guard on n_workers == 0
    # alone would let -1 reach multiprocessing.Pool and fail opaquely there.
    with pytest.raises(ValueError, match='at least 1'):
        init_mod.sample_from_bounds(
            output='out',
            ref_config='ref.toml',
            params={'a': [0.0, 1.0]},
            observables={'obs': 1.0},
            nsamp=2,
            seed=1,
            n_workers=-1,
            failure_codes=[],
        )


@pytest.mark.unit
def test_sample_from_bounds_caps_workers_and_saves(monkeypatch, tmp_path):
    """``sample_from_bounds`` caps the worker pool at ``cpu_count - 1``
    (3 in this test, with cpu_count=4 mocked) regardless of the user's
    request, uses Halton sequences for the initial design, and saves
    the resulting (X, Y) dataset to ``init.csv``.
    """
    captured = {}

    class FakeHalton:
        def __init__(self, d, rng, scramble):
            captured['dims'] = d
            captured['scramble'] = scramble

        def random(self, n):
            return np.array([[0.1], [0.9]])[:n]

    class FakePool:
        def __init__(self, processes, initializer=None, initargs=()):
            captured['processes'] = processes
            captured['initializer'] = initializer

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap_async(self, func, args):
            captured['task_count'] = len(args)
            results = [torch.tensor([[0.2]], dtype=torch.double) for _ in args]

            class _AsyncResult:
                def get(self, timeout=None):
                    captured['pool_timeout'] = timeout
                    return results

            return _AsyncResult()

    monkeypatch.setattr(init_mod.os, 'cpu_count', lambda: 4)
    monkeypatch.setattr(init_mod, 'Halton', FakeHalton)
    monkeypatch.setattr(init_mod, 'Pool', FakePool)
    monkeypatch.setattr(
        init_mod, 'get_proteus_directories', lambda _output: {'output': str(tmp_path)}
    )

    def fake_save_dataset_csv(X, Y, fpath):
        captured['saved_shape'] = (tuple(X.shape), tuple(Y.shape))
        captured['saved_path'] = fpath

    monkeypatch.setattr(init_mod, 'save_dataset_csv', fake_save_dataset_csv)

    n = init_mod.sample_from_bounds(
        output='out',
        ref_config='ref.toml',
        params={'a': [0.0, 1.0]},
        observables={'obs': 1.0},
        nsamp=2,
        seed=11,
        n_workers=10,
        failure_codes=[1, 10],
    )

    assert n == 2
    assert captured['processes'] == 3
    assert captured['task_count'] == 2
    # Each pool worker is handed the batch's stop signal.
    assert captured['initializer'] is init_mod._init_pool_worker
    assert captured['saved_shape'] == ((2, 1), (2, 1))
    assert captured['saved_path'].endswith('init.csv')
    # The batch is bounded so a wedged worker cannot hang it indefinitely:
    # the default per-child timeout (6 h) yields a positive, finite pool cap.
    assert captured['pool_timeout'] is not None
    assert captured['pool_timeout'] > 0


@pytest.mark.unit
def test_a_failed_initial_sample_stops_the_samples_not_yet_started(monkeypatch):
    """Under `abort_on_failure` the first failed run fails the whole initial
    batch, so the samples still queued must not each run a simulation first.
    The failing sample sets the batch's stop signal, and a sample that finds it
    set returns without building an objective or running anything.
    """
    stop = threading.Event()
    monkeypatch.setattr(init_mod, '_stop', None)
    init_mod._init_pool_worker(stop)
    args = {'parameters': {'a': [0.0, 1.0]}, 'observables': {'obs': 1.0}}
    args |= {'ref_config': 'ref.toml', 'output': 'out', 'failure_codes': []}
    x = torch.tensor([[0.5]], dtype=torch.double)

    # Discrimination: a sample that succeeds returns its score and leaves the
    # signal clear, so the stop below is the failure's doing.
    score = torch.tensor([[1.25]], dtype=torch.double)
    monkeypatch.setattr(init_mod, 'prot_builder', lambda **_kw: lambda _x: score)
    assert init_mod.f_aug(x, 0, args).item() == pytest.approx(1.25)
    assert not stop.is_set()

    def _failing(_x):
        raise ProteusRunFailure(reason='r', worker=-1, iter=1, out_dir='/o', status=21)

    monkeypatch.setattr(init_mod, 'prot_builder', lambda **_kw: _failing)
    with pytest.raises(ProteusRunFailure):
        init_mod.f_aug(x, 1, args)
    assert stop.is_set()

    # A later sample runs nothing: building its objective would fail the test.
    monkeypatch.setattr(
        init_mod, 'prot_builder', lambda **_kw: pytest.fail('a skipped sample ran')
    )
    assert init_mod.f_aug(x, 2, args) is None

    # Edge case: called outside a pool, with no signal installed, a failure
    # propagates as before and nothing is skipped.
    monkeypatch.setattr(init_mod, '_stop', None)
    monkeypatch.setattr(init_mod, 'prot_builder', lambda **_kw: _failing)
    with pytest.raises(ProteusRunFailure):
        init_mod.f_aug(x, 3, args)


@pytest.mark.unit
def test_real_halton_accepts_rng_keyword_and_is_deterministic():
    """Check halton sampler accepts rng and behaves pseudo-deterministically.

    ``sample_from_bounds`` constructs ``Halton(d=..., rng=..., scramble=True)``.
    The ``rng=`` keyword replaced the deprecated ``seed=`` in scipy 1.15.0
    """
    import scipy
    from packaging.version import Version
    from scipy.stats.qmc import Halton

    # Version floor recorded in pyproject.toml (scipy>=1.15.0).
    assert Version(scipy.__version__) >= Version('1.15.0')

    # Set up sampler with fixed seed of 42 (answer to life, universe, and everything).
    dims = 3
    nsamp = 5
    sampler = Halton(d=dims, rng=np.random.default_rng(42), scramble=True)
    x = np.asarray(sampler.random(nsamp))

    # Shape and unit-hypercube bounds: Halton draws live in [0, 1).
    assert x.shape == (nsamp, dims)
    assert x.min() >= 0.0
    assert x.max() < 1.0

    # Determinism guard. This rules out a silently non-seeded sampler.
    x_same = np.asarray(
        Halton(d=dims, rng=np.random.default_rng(42), scramble=True).random(nsamp)
    )
    x_diff = np.asarray(
        Halton(d=dims, rng=np.random.default_rng(7), scramble=True).random(nsamp)
    )
    np.testing.assert_allclose(x, x_same, rtol=0.0, atol=0.0)
    assert not np.allclose(x, x_diff)
