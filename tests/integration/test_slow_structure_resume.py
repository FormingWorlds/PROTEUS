"""Structure re-solves of a resumed run happen at the rows of the run it continues.

Source under test: the structure-trigger references (``struct_ref_*``) that
``update_structure_from_interior`` (``src/proteus/interior_energetics/wrapper.py``) keeps in the
helpfile row, which carries them to the next step and to a resume.

A run of ``tests/integration/zalmoxis_aragog_calliope.toml`` with a fixed 50 yr step and a
structure update at most every 100 yr runs to 302 yr in one go. The same run stops at 152 yr
and resumes to 302 yr. A copy of the stopped run whose last row takes its own time, T_magma and
Phi as the references, as a resume that did not keep them would, resumes as well.

Invariants asserted:

- The run in one go re-solves before the stop and at least twice after it.
- The resumed run has the rows and the re-solve rows of the run in one go.
- The copy with the row as reference re-solves at other rows (discrimination).
- T_magma stays positive and Phi_global in [0, 1] in the resumed run.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest

pytest.importorskip('zalmoxis')

# Four Zalmoxis-Aragog-CALLIOPE runs, 25 rows in all; the nightly runner is slower than Habrok.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(7200)]

CONFIG = pathlib.Path(__file__).parent / 'zalmoxis_aragog_calliope.toml'
REF_KEYS = ('struct_ref_time', 'struct_ref_T_magma', 'struct_ref_Phi')

# After the init rows (0, 1, 2 yr) the steps are 50 yr, and the 100 yr ceiling re-solves at 102, 202
# and 302 yr. A stop at 152 yr keeps the step grid; a resume that takes its row as the reference
# moves the next re-solve to 252 yr.
DT_YR = 50.0
CEILING_YR = 100.0
T_STOP_YR = 152.0
T_END_YR = 302.0


def _runner(outdir, tmax):
    from proteus import Proteus

    runner = Proteus(config_path=str(CONFIG))
    runner.config.params.out.path = str(outdir)
    z = runner.config.interior_struct.zalmoxis
    z.update_interval, z.update_min_interval = CEILING_YR, 0.0
    runner.init_directories()
    dt = runner.config.params.dt
    dt.method, dt.initial, dt.minimum, dt.maximum = 'maximum', DT_YR, 5.0, DT_YR
    runner.config.params.stop.time.minimum = 100.0
    runner.config.params.stop.time.maximum = tmax
    out = runner.config.params.out
    out.plot_mod, out.write_mod, out.dt_write_rel, out.archive_mod = None, 1, 0.0, 'none'
    return runner


def _resolve_times(hf):
    """Model times [yr] of the re-solves: the rows whose structure reference is their own time."""
    t, ref = hf['Time'].to_numpy(), hf['struct_ref_time'].to_numpy()
    return t[(t > 0) & np.isclose(ref, t, rtol=1e-12, atol=0)]


def _take_the_row_as_reference(outdir):
    """Set the trigger references of the last (tab-separated) helpfile row to its own time,
    T_magma and Phi_global."""
    path = outdir / 'runtime_helpfile.csv'
    lines = path.read_text().splitlines()
    header = lines[0].split('\t')
    values = lines[-1].split('\t')
    for key, own in zip(REF_KEYS, ('Time', 'T_magma', 'Phi_global')):
        values[header.index(key)] = values[header.index(own)]
    lines[-1] = '\t'.join(values)
    path.write_text('\n'.join(lines) + '\n')


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp('structure_resume')
    whole = _runner(base / 'whole', T_END_YR)
    whole.start(resume=False, offline=True)

    _runner(base / 'resumed', T_STOP_YR).start(resume=False, offline=True)
    shutil.copytree(base / 'resumed', base / 'row_refs')
    _take_the_row_as_reference(base / 'row_refs')

    resumed = _runner(base / 'resumed', T_END_YR)
    resumed.start(resume=True, offline=True)
    row_refs = _runner(base / 'row_refs', T_END_YR)
    row_refs.start(resume=True, offline=True)
    return {
        'whole': whole.hf_all.reset_index(drop=True),
        'resumed': resumed.hf_all.reset_index(drop=True),
        'row_refs': row_refs.hf_all.reset_index(drop=True),
    }


@pytest.mark.physics_invariant
def test_resumed_run_resolves_the_structure_at_the_rows_of_the_run_in_one_go(runs):
    """Zalmoxis and Aragog with CALLIOPE outgassing: the run stopped at 152 yr and resumed
    re-solves at the rows of the run in one go; a resume from the row's own state does not."""
    whole, resumed, row_refs = runs['whole'], runs['resumed'], runs['row_refs']
    t_whole = _resolve_times(whole)
    assert (t_whole < T_STOP_YR).any()
    assert (t_whole > T_STOP_YR).sum() >= 2

    np.testing.assert_allclose(resumed['Time'], whole['Time'], rtol=1e-12)
    np.testing.assert_allclose(_resolve_times(resumed), t_whole, rtol=1e-12)
    np.testing.assert_allclose(resumed['struct_ref_time'], whole['struct_ref_time'], rtol=1e-12)
    # After the stop a resumed Aragog run reproduces T and Phi within STATE_RTOL of
    # test_slow_aragog_resume.py; before it the two fresh runs differ by up to 2.4e-5 in T.
    after = (whole['Time'] > T_STOP_YR).to_numpy()
    for key in ('struct_ref_T_magma', 'struct_ref_Phi'):
        np.testing.assert_allclose(resumed[key][after], whole[key][after], rtol=1e-5)

    t_row_refs = _resolve_times(row_refs)
    assert len(t_row_refs) != len(t_whole) or not np.allclose(t_row_refs, t_whole)

    assert (resumed['T_magma'][after] > 0).all()
    assert resumed['Phi_global'][after].between(0.0, 1.0).all()
