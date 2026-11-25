"""Structure re-solves of a resumed run happen at the rows of the run it continues.

Source under test: the structure-trigger references that ``Proteus`` stores in each helpfile
row (``store_structure_trigger_state``) and restores on resume
(``restore_structure_trigger_state``, ``src/proteus/interior_energetics/wrapper.py``).

A run of ``tests/integration/zalmoxis_aragog_calliope.toml`` with a fixed 50 yr step and a
structure update at most every 150 yr runs to 500 yr in one go. The same run stops at 252 yr
and resumes to 500 yr. A copy of the stopped run with the references of its last row set to
0, which is what a resume read before they were stored, resumes as well.

Invariants asserted:

- The run in one go re-solves before the stop and at least twice after it.
- The resumed run has the rows and the re-solve rows of the run in one go.
- The copy without references re-solves at other rows (discrimination).
- T_magma stays positive and Phi_global in [0, 1] in the resumed run.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest

pytest.importorskip('zalmoxis')

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

CONFIG = pathlib.Path(__file__).parent / 'zalmoxis_aragog_calliope.toml'
REF_KEYS = ('struct_ref_time', 'struct_ref_T_magma', 'struct_ref_Phi')

# After the init rows (0, 1, 2 yr) the steps are 50 yr, and the 150 yr ceiling re-solves at 152, 302
# and 452 yr. A stop at 252 yr keeps the step grid; a resume that takes its row as the reference
# moves the next re-solve to 402 yr.
DT_YR = 50.0
CEILING_YR = 150.0
T_STOP_YR = 252.0
T_END_YR = 500.0


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


def _drop_references(outdir):
    """Set the trigger references of the last (tab-separated) helpfile row to 0, as an older
    helpfile reads."""
    path = outdir / 'runtime_helpfile.csv'
    lines = path.read_text().splitlines()
    header = lines[0].split('\t')
    values = lines[-1].split('\t')
    for key in REF_KEYS:
        values[header.index(key)] = '0.0'
    lines[-1] = '\t'.join(values)
    path.write_text('\n'.join(lines) + '\n')


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp('structure_resume')
    whole = _runner(base / 'whole', T_END_YR)
    whole.start(resume=False, offline=True)

    _runner(base / 'resumed', T_STOP_YR).start(resume=False, offline=True)
    shutil.copytree(base / 'resumed', base / 'no_refs')
    _drop_references(base / 'no_refs')

    resumed = _runner(base / 'resumed', T_END_YR)
    resumed.start(resume=True, offline=True)
    no_refs = _runner(base / 'no_refs', T_END_YR)
    no_refs.start(resume=True, offline=True)
    return {
        'whole': whole.hf_all.reset_index(drop=True),
        'resumed': resumed.hf_all.reset_index(drop=True),
        'no_refs': no_refs.hf_all.reset_index(drop=True),
    }


@pytest.mark.physics_invariant
def test_resumed_run_resolves_the_structure_at_the_rows_of_the_run_in_one_go(runs):
    """Zalmoxis and Aragog with CALLIOPE outgassing: the run stopped at 252 yr and resumed
    re-solves at the rows of the run in one go; the resume without references does not."""
    whole, resumed, no_refs = runs['whole'], runs['resumed'], runs['no_refs']
    t_whole = _resolve_times(whole)
    assert (t_whole < T_STOP_YR).any()
    assert (t_whole > T_STOP_YR).sum() >= 2

    np.testing.assert_allclose(resumed['Time'], whole['Time'], rtol=1e-12)
    np.testing.assert_allclose(_resolve_times(resumed), t_whole, rtol=1e-12)
    np.testing.assert_allclose(resumed['struct_ref_time'], whole['struct_ref_time'], rtol=1e-12)
    # A resumed Aragog run reproduces T and Phi within STATE_RTOL of test_slow_aragog_resume.py.
    for key in ('struct_ref_T_magma', 'struct_ref_Phi'):
        np.testing.assert_allclose(resumed[key], whole[key], rtol=1e-5)

    t_no_refs = _resolve_times(no_refs)
    assert len(t_no_refs) != len(t_whole) or not np.allclose(t_no_refs, t_whole)

    after = (resumed['Time'] > T_STOP_YR).to_numpy()
    assert (resumed['T_magma'][after] > 0).all()
    assert resumed['Phi_global'][after].between(0.0, 1.0).all()
