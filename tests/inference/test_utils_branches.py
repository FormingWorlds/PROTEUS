"""Branch coverage for ``proteus.inference.utils``.

Targets the previously untested code paths in ``load_dataset_csv``,
``get_obs`` (including the P_surf-collapsed escaped-atmosphere branch),
and ``print_results``. ``print_results`` orchestrates several file
reads and is exercised here against a small synthetic workspace
written to ``tmp_path``.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

# The Bayesian-optimisation stack ships as the optional `inference` extra,
# which installs all three together. Guarding the whole stack keeps a
# partial environment skipping rather than failing collection.
torch = pytest.importorskip('torch')
pytest.importorskip('botorch')
pytest.importorskip('gpytorch')

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# load_dataset_csv: happy path
# ---------------------------------------------------------------------------


def test_load_dataset_csv_round_trips_x_and_y_tensors(tmp_path):
    """Saving X/Y via ``save_dataset_csv`` and reading back via
    ``load_dataset_csv`` must produce tensors numerically identical to
    the originals. Discrimination: order of x_* columns is determined
    by the integer suffix; pinning two different values across x_0/x_1
    guards against a column-order regression.
    """
    from proteus.inference.utils import load_dataset_csv, save_dataset_csv

    X = torch.tensor([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    Y = torch.tensor([[10.0], [20.0], [30.0]])
    csv_path = tmp_path / 'dataset.csv'

    save_dataset_csv(X, Y, str(csv_path))
    data = load_dataset_csv(str(csv_path))

    np.testing.assert_allclose(data['X'].numpy(), X.numpy())
    np.testing.assert_allclose(data['Y'].numpy(), Y.numpy())
    # Discrimination: shape preserved exactly (3, 2) and (3, 1).
    assert data['X'].shape == X.shape
    assert data['Y'].shape == Y.shape


# ---------------------------------------------------------------------------
# get_obs: normal vs escaped-atmosphere branch
# ---------------------------------------------------------------------------


def _write_runtime_csv(path, *, p_surf, atm_kg_per_mol=0.02897, h2o_vmr=0.5, co2_vmr=0.5):
    """Build a one-row whitespace-delimited helpfile CSV that
    ``get_obs`` can parse. Only the columns referenced by the function
    need to be present.
    """
    # The reader uses ``delimiter=r'\\s+'`` (whitespace), so write
    # space-separated values with a single header row.
    cols = [
        'Time',
        'P_surf',
        'atm_kg_per_mol',
        'H2O_vmr',
        'CO2_vmr',
        'H2_vmr',
        'CO_vmr',
        'CH4_vmr',
        'N2_vmr',
        'NH3_vmr',
        'S2_vmr',
        'SO2_vmr',
        'H2S_vmr',
        'O2_vmr',
        'OCS_vmr',
        'HCN_vmr',
        'C2H2_vmr',
        'C2H6_vmr',
        'HCl_vmr',
        'HF_vmr',
        'Cl2_vmr',
        'NO_vmr',
        'NO2_vmr',
        'N2O_vmr',
        'CN_vmr',
        'NCO_vmr',
        'CHO_vmr',
        'CH2O_vmr',
        'C2N2_vmr',
        'CH3OH_vmr',
    ]
    vals = [1.0e8, p_surf, atm_kg_per_mol, h2o_vmr, co2_vmr] + [0.0] * (len(cols) - 5)
    with open(path, 'w') as fh:
        fh.write(' '.join(cols) + '\n')
        fh.write(' '.join(f'{v:.6e}' for v in vals) + '\n')


def test_get_obs_returns_observable_subset_under_normal_atmosphere(tmp_path):
    """For a normal-atmosphere row (P_surf well above the 1e-30 floor)
    ``get_obs`` returns the requested observables straight from the
    CSV. Discrimination: P_surf must equal what we wrote, not the
    zero-fallback used in the escaped branch.
    """
    from proteus.inference.utils import get_obs

    csv = tmp_path / 'runtime_helpfile.csv'
    _write_runtime_csv(csv, p_surf=1.5e7, atm_kg_per_mol=0.030, h2o_vmr=0.42, co2_vmr=0.58)

    result = get_obs(str(csv), observables=['P_surf', 'atm_kg_per_mol', 'H2O_vmr'])

    assert result['P_surf'] == pytest.approx(1.5e7)
    assert result['atm_kg_per_mol'] == pytest.approx(0.030)
    assert result['H2O_vmr'] == pytest.approx(0.42)


def test_get_obs_zeroes_vmr_and_mmw_when_atmosphere_has_escaped(tmp_path):
    """When P_surf is below 1e-30 the atmosphere has effectively
    escaped; ``get_obs`` overwrites every ``*_vmr`` and the mean
    molecular weight to zero so downstream consumers do not divide by
    a vanishing pressure. Discrimination: the H2O_vmr written to disk
    (0.5) must be overwritten by 0.0 in the returned series.
    """
    from proteus.inference.utils import get_obs

    csv = tmp_path / 'runtime_helpfile.csv'
    _write_runtime_csv(csv, p_surf=1.0e-40, h2o_vmr=0.5, co2_vmr=0.5)

    result = get_obs(str(csv), observables=['atm_kg_per_mol', 'H2O_vmr', 'CO2_vmr'])

    assert result['atm_kg_per_mol'] == 0.0
    assert result['H2O_vmr'] == 0.0
    assert result['CO2_vmr'] == 0.0


# ---------------------------------------------------------------------------
# print_results
# ---------------------------------------------------------------------------


def _make_worker_dir(root, worker, iteration, *, obs_value, param_value):
    """Build the worker dir layout expected by ``print_results``:
    ``<output>/workers/w_<worker>/i_<iteration>/`` containing
    ``runtime_helpfile.csv`` and ``init_coupler.toml``.
    """
    wdir = root / 'workers' / f'w_{worker}' / f'i_{iteration}'
    wdir.mkdir(parents=True)
    csv = wdir / 'runtime_helpfile.csv'
    _write_runtime_csv(csv, p_surf=1.0e7, h2o_vmr=obs_value, co2_vmr=0.5)
    toml_path = wdir / 'init_coupler.toml'
    toml_path.write_text(f'[planet]\nmass_tot = {param_value}\n')
    return wdir


def test_print_results_returns_best_input_toml_path_and_logs_summary(tmp_path, caplog):
    """``print_results`` selects the entry with the maximum objective
    (excluding the first ``n_init`` entries) and returns the path to
    that case's input TOML. Discrimination: with the second non-init
    entry holding the largest Y, the returned path must reference
    iteration 1 (not the init or the larger-index entry).
    """
    from proteus.inference.utils import print_results

    # Build a workspace with three runs: indices 0..2. Mark the first as
    # the initial guess (n_init=1) so only entries 1 and 2 are eligible
    # for the best-case selection. Entry 1 carries the largest Y and
    # therefore must be picked.
    _make_worker_dir(tmp_path, worker=0, iteration=0, obs_value=0.1, param_value=0.5)
    best_dir = _make_worker_dir(tmp_path, worker=0, iteration=1, obs_value=0.9, param_value=1.0)
    _make_worker_dir(tmp_path, worker=0, iteration=2, obs_value=0.5, param_value=1.5)

    D = {
        'X': torch.tensor([[0.0], [1.0], [0.5]]),
        'Y': torch.tensor([[-1.0], [5.0], [2.0]]),
    }
    logs = [
        {'worker': 0, 'task_id': 0},
        {'worker': 0, 'task_id': 1},
        {'worker': 0, 'task_id': 2},
    ]
    config = {
        'observables': {'H2O_vmr': 0.9},
        'parameters': {'planet.mass_tot': [0.5, 1.5]},
    }

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.utils'):
        result = print_results(D, logs, config, str(tmp_path), n_init=1)

    # Discrimination: result must point at iteration 1's input TOML.
    assert str(result).endswith('i_1/init_coupler.toml')
    assert str(best_dir / 'init_coupler.toml') == str(result)
    # The summary log must announce the best-case step number relative
    # to the initial-guess offset (i_opt - n_init + 1 = 1).
    assert any('Best case was step 1' in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# get_kernel: Matern branches and error contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_kernel_mat12_returns_matern_with_nu_half():
    """get_kernel('MAT1/2', d) builds a once-differentiable Matern kernel (nu=0.5)."""
    from gpytorch.kernels import MaternKernel

    from proteus.inference.utils import get_kernel

    result = get_kernel('MAT1/2', d=2)

    assert isinstance(result, MaternKernel)
    assert result.nu == pytest.approx(0.5)
    assert result.nu != pytest.approx(1.5)
    assert result.nu != pytest.approx(2.5)

    # ARD: lengthscale shape must be (1, d) for d=2 dimensions.
    assert result.lengthscale.shape[-1] == 2


@pytest.mark.unit
def test_get_kernel_mat32_returns_matern_with_nu_three_halves():
    """get_kernel('MAT3/2', d) builds a once-mean-square-differentiable Matern (nu=1.5).

    nu=1.5 is the default used in the vanilla-BO kernel prior (Hvarfner 2024).
    A regression that wired this branch to nu=0.5 or nu=2.5 would change
    the inductive bias of the surrogate across the whole BO campaign.
    """
    from gpytorch.kernels import MaternKernel

    from proteus.inference.utils import get_kernel

    result = get_kernel('MAT3/2', d=3)

    assert isinstance(result, MaternKernel)
    assert result.nu == pytest.approx(1.5)
    # Discrimination: wrong-nu branches differ by exactly 1.0 or 0.5.
    assert abs(result.nu - 0.5) > 0.5
    assert abs(result.nu - 2.5) > 0.5
    assert result.lengthscale.shape[-1] == 3


@pytest.mark.unit
def test_get_kernel_mat52_returns_matern_with_nu_five_halves():
    """get_kernel('MAT5/2', d) builds a twice-mean-square-differentiable Matern (nu=2.5).

    nu=2.5 produces smoother sample paths than nu=1.5 or 0.5. Checking the
    stored nu guards against a copy-paste error between the MAT3/2 and MAT5/2
    branches (using nu=1.5 instead of 2.5), which would degrade optimisation
    quality on smooth physics objectives without any error.
    """
    from gpytorch.kernels import MaternKernel

    from proteus.inference.utils import get_kernel

    result = get_kernel('MAT5/2', d=1)

    assert isinstance(result, MaternKernel)
    assert result.nu == pytest.approx(2.5)
    # Discrimination: a MAT3/2 regression would land at 1.5, more than 0.5 below.
    assert result.nu > 2.0
    assert result.lengthscale.shape[-1] == 1


@pytest.mark.unit
def test_get_kernel_raises_for_unknown_kernel_name():
    """get_kernel raises ValueError for names outside the documented set.

    The error must fire before any kernel object is built, so no partial
    state is returned. An unknown kernel name from a mis-configured
    infer.toml must surface as a clear diagnostic rather than a silent
    None return or a cryptic GPyTorch exception.

    Discrimination: each of the four valid names is also exercised (implicitly
    by the three tests above) to confirm the error fires only on invalid input.
    """
    from proteus.inference.utils import get_kernel

    with pytest.raises(ValueError, match='Unknown kernel'):
        get_kernel('POLY', d=2)

    with pytest.raises(ValueError, match='Unknown kernel'):
        get_kernel('', d=2)

    # Edge: case-sensitive — 'rbf' is not 'RBF'.
    with pytest.raises(ValueError, match='Unknown kernel'):
        get_kernel('rbf', d=2)


# ---------------------------------------------------------------------------
# Accounting for the evaluations that failed
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_print_results_counts_unscored_runs_and_refuses_a_study_with_no_fit(tmp_path, caplog):
    """Evaluations that failed, and those that completed on an excluded status,
    both carry the failure score rather than a fit quality, so the summary says
    how many of them there were. The objective value alone cannot tell the two
    apart, so the wording covers both. When no optimisation evaluation produced
    a fit quality there is no best fit at all, and the summary stops rather than
    reporting the least-bad run as an inference result.
    """
    from proteus.inference.objective import BAD_OBJ_VALUE
    from proteus.inference.utils import print_results

    _make_worker_dir(tmp_path, worker=0, iteration=0, obs_value=0.1, param_value=0.5)
    _make_worker_dir(tmp_path, worker=0, iteration=1, obs_value=0.9, param_value=1.0)
    best_dir = _make_worker_dir(tmp_path, worker=0, iteration=2, obs_value=0.5, param_value=1.5)

    logs = [
        {'worker': 0, 'task_id': 0},
        {'worker': 0, 'task_id': 1},
        {'worker': 0, 'task_id': 2},
    ]
    config = {
        'observables': {'H2O_vmr': 0.9},
        'parameters': {'planet.mass_tot': [0.5, 1.5]},
    }

    # One of the two optimisation evaluations failed; the other is still the
    # best fit and must be reported normally.
    D = {
        'X': torch.tensor([[0.0], [1.0], [0.5]]),
        'Y': torch.tensor([[-1.0], [BAD_OBJ_VALUE], [2.0]]),
    }
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.inference.utils'):
        result = print_results(D, logs, config, str(tmp_path), n_init=1)
    assert str(best_dir / 'init_coupler.toml') == str(result)
    assert any(
        '1 of 2 optimisation evaluations carry the failure score' in r.message
        for r in caplog.records
    )

    # Discrimination: the same study with no failure score present reports no
    # count, so the message above tracks the data and is not emitted always.
    caplog.clear()
    D_clean = {
        'X': torch.tensor([[0.0], [1.0], [0.5]]),
        'Y': torch.tensor([[-1.0], [1.0], [2.0]]),
    }
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.inference.utils'):
        print_results(D_clean, logs, config, str(tmp_path), n_init=1)
    assert not [r for r in caplog.records if 'carry the failure score' in r.message]

    # No optimisation evaluation produced a fit quality: nothing to report.
    D_dead = {
        'X': torch.tensor([[0.0], [1.0], [0.5]]),
        'Y': torch.tensor([[-1.0], [BAD_OBJ_VALUE], [BAD_OBJ_VALUE]]),
    }
    with pytest.raises(
        RuntimeError, match='None of the 2 optimisation evaluations produced a fit quality'
    ):
        print_results(D_dead, logs, config, str(tmp_path), n_init=1)


@pytest.mark.unit
def test_summarise_failures_tabulates_causes_and_flags_a_mostly_failed_study(tmp_path, caplog):
    """The end-of-study tally turns the per-run records into one table and one
    breakdown by cause, and escalates to a warning once most of the study
    failed. Without the escalation, a posterior built on a handful of real
    evaluations reads the same as one built on all of them.
    """
    from proteus.inference.objective import ProteusRunFailure, record_failure
    from proteus.inference.utils import summarise_failures

    # Two runs that died the same way and one that died differently, so the
    # breakdown has something to group.
    for worker, status in ((0, 21), (1, 21), (2, 24)):
        record_failure(
            tmp_path,
            ProteusRunFailure(
                reason='the simulator exited with an error',
                worker=worker,
                iter=0,
                out_dir=f'/study/workers/w_{worker}/i_0',
                exit_code=1,
                status=status,
                parameters={'planet.mass_tot': 1.0 + worker},
            ),
        )

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.utils'):
        n_failed = summarise_failures(str(tmp_path), n_attempted=20)

    assert n_failed == 3
    messages = '\n'.join(r.message for r in caplog.records)
    assert '3 of 20 evaluations' in messages
    # Grouped by cause, so two runs that died the same way count as one line.
    assert 'Interior model' in messages
    # Below the escalation threshold (3/20 = 15%), the tally is reported but
    # not warned about.
    assert not [r for r in caplog.records if r.levelname == 'WARNING']

    # The table carries the swept parameter alongside the diagnosis, so the
    # failing region can be located without opening each run folder.
    table = pd.read_csv(tmp_path / 'failures.csv')
    assert len(table) == 3
    assert list(table['worker']) == [0, 1, 2]
    assert sorted(table['status']) == [21, 21, 24]
    assert table['planet.mass_tot'].max() == pytest.approx(3.0)

    # Discrimination: the same three failures against a smaller study cross the
    # threshold and are warned about. A tally without the escalation would log
    # identically in both cases.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.utils'):
        summarise_failures(str(tmp_path), n_attempted=4)
    warnings = [r for r in caplog.records if r.levelname == 'WARNING']
    assert len(warnings) == 1
    # The count of evaluations that are real is what the reader needs.
    assert '1 real evaluations' in warnings[0].message


@pytest.mark.unit
def test_summarise_failures_counts_excluded_outcomes_apart_from_failures(tmp_path, caplog):
    """A run that completed on a status the study excludes is tallied, but not
    as a fault. Folding the two together would tell the user that a study whose
    runs all reached their clock limit, exactly as configured, is a study full
    of broken simulations.
    """
    from proteus.inference.objective import (
        CATEGORY_EXCLUDED,
        ProteusRunFailure,
        record_failure,
    )
    from proteus.inference.utils import summarise_failures

    record_failure(
        tmp_path,
        ProteusRunFailure(
            reason='the simulator exited with an error',
            worker=0,
            iter=0,
            out_dir='/study/workers/w_0/i_0',
            exit_code=1,
            status=21,
            parameters={'planet.mass_tot': 1.0},
        ),
    )
    for worker in (1, 2):
        record_failure(
            tmp_path,
            ProteusRunFailure(
                reason='completed on a status this study excludes',
                worker=worker,
                iter=0,
                out_dir=f'/study/workers/w_{worker}/i_0',
                exit_code=0,
                status=11,
                parameters={'planet.mass_tot': 1.0 + worker},
                category=CATEGORY_EXCLUDED,
            ),
        )

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.utils'):
        n_unscored = summarise_failures(str(tmp_path), n_attempted=20)

    # Both kinds are unscored, so both count toward how much of the study was
    # real, but the breakdown names them apart.
    assert n_unscored == 3
    messages = '\n'.join(r.message for r in caplog.records)
    assert '1 did not produce a usable result' in messages
    assert '2 completed on a status this study excludes' in messages
    # The clock-limit outcome is labelled in the cause table rather than being
    # listed beside the interior-model error as if it were one.
    assert 'Completed (maximum clock runtime) [excluded]' in messages
    assert 'Error (Interior model) [excluded]' not in messages

    # Carried into the table too, so the excluded rows can be filtered out when
    # looking for the region that actually breaks the simulator.
    table = pd.read_csv(tmp_path / 'failures.csv')
    assert sorted(table['category']) == ['excluded', 'excluded', 'failure']
    assert sorted(table.loc[table['category'] == 'excluded', 'status']) == [11, 11]


@pytest.mark.unit
def test_summarise_failures_reports_a_clean_study_without_writing_a_table(tmp_path, caplog):
    """A study in which nothing failed says so and writes no table. An empty
    failures.csv would suggest the accounting had run and found nothing to
    say about a study that in fact had nothing to report.
    """
    from proteus.inference.utils import summarise_failures

    with caplog.at_level(logging.INFO, logger='fwl.proteus.inference.utils'):
        n_failed = summarise_failures(str(tmp_path), n_attempted=12)

    assert n_failed == 0
    assert not (tmp_path / 'failures.csv').exists()
    messages = '\n'.join(r.message for r in caplog.records)
    assert 'none' in messages and '12 evaluations' in messages
    assert not [r for r in caplog.records if r.levelname in ('WARNING', 'ERROR')]
