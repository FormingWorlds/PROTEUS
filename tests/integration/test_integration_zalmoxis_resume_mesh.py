"""Integration test: regenerating a resumed run's Zalmoxis mesh from real EOS physics.

Resuming a run at an earlier row must regenerate ``data/zalmoxis_output.dat``
against that row's own boundary state, not leave the directory's existing
mesh file (written for whichever row the previous leg last reached) in
place. Aragog's own EOS-radius-range guard
(``aragog.solver.entropy_solver._validate_eos_radius_range``) rejects a
mesh whose top radius drifts from the resumed row's ``R_int`` by more than
about a metre, so any mismatch between the resumed row and the on-disk
mesh surfaces as a hard crash rather than a silently wrong structure.

This test drives ``zalmoxis_solver`` with the real EOS and real
configuration of an archived reference run, resumed at an earlier row than
the run's own final state, and checks the regenerated mesh against that
row with ``validate_zalmoxis_output_schema``. The radius half of that check
is guaranteed to pass once the solver call returns without raising, since
``zalmoxis_solver`` mutates ``hf_row['R_int']`` to its own converged value
and applies the identical radius check to itself before returning. The
mass half is not: it omits ``mantle_mass_ref``, so it recomputes the
mantle mass from the written file's radius/density columns rather than
reusing the solver's in-memory accumulator, which is a separate
computation from the one the solver ran on itself. It is a real-physics
run, not a mocked one: no Zalmoxis or EOS call is patched.

Scope. The fixture under ``tests/data/integration/zalmoxis_resume_mesh/``
is a single row and mesh file taken from a real archived run, trimmed down
from the run's full helpfile and reference config; its config's
``accretion`` table, which belonged only to a checkout of an in-progress
module the archived run was captured under, is stripped so the fixture
loads against the current config schema. Its helpfile row also predates
two escape-tracking columns the current schema adds
(``esc_clamp_frac``, ``esc_step_kg``), backfilled here with zero since
neither is read by the interior solve this test exercises. A future
schema change can still desync the fixture; that is treated as a skip
rather than a failure, since it reflects which schema the fixture
predates rather than a defect in mesh regeneration. It exercises
``zalmoxis_solver`` directly on a
row picked to reproduce a truncated-replay resume scenario, not through a
full ``Proteus.start(resume=True)`` call, so it covers mesh regeneration in
isolation. It does not cover the surrounding resume orchestration: real
single-crash resume runs always restart from the helpfile's last row
(``self.hf_row = self.hf_all.iloc[-1].to_dict()`` in ``proteus.py``), which
``test_integration_resume.py`` exercises end to end on the dummy backends,
but that file's own backends write no Zalmoxis mesh and do not exercise
this regeneration step either. ``select_resumable_snapshot``'s row-trimming
behaviour is unit-tested in ``tests/utils/test_coupler.py``; whether a
genuine mid-run crash can leave that function selecting a row earlier than
the one ``zalmoxis_output.dat`` was last written for is not covered by any
test at present.

Documentation: For testing standards, see:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from proteus.config import UnknownConfigKeyError, read_config_object
from proteus.interior_struct.zalmoxis import validate_zalmoxis_output_schema, zalmoxis_solver
from proteus.utils.coupler import ReadHelpfileFromCSV

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

_REFERENCE_RUN_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'integration', 'zalmoxis_resume_mesh'
)
_CONFIG_PATH = os.path.join(_REFERENCE_RUN_DIR, 'init_coupler.toml')
_HELPFILE_PATH = os.path.join(_REFERENCE_RUN_DIR, 'runtime_helpfile.csv')
_STALE_MESH_PATH = os.path.join(_REFERENCE_RUN_DIR, 'zalmoxis_output.dat')

# The fixture helpfile carries only the resumed row, at index 0; the run's
# own row index in the archived helpfile this was trimmed from was 200,
# well earlier than the run's own final row, to reproduce a replay resumed
# from a stale on-disk mesh. It is not where this run's own resume path
# would restart from (that is always the helpfile's last row).
_RESUMED_ROW_INDEX = 0
_RESUMED_ROW_TIME_YR = 5.0e5


@pytest.mark.skipif(
    not (
        os.path.isfile(_CONFIG_PATH)
        and os.path.isfile(_HELPFILE_PATH)
        and os.path.isfile(_STALE_MESH_PATH)
    ),
    reason='fixture missing under tests/data/integration/zalmoxis_resume_mesh/',
)
def test_regenerated_mesh_matches_resumed_row_and_stale_mesh_does_not(tmp_path):
    """Zalmoxis regenerates a mesh consistent with the resumed row's own state.

    ``zalmoxis_solver`` mutates ``hf_row['R_int']`` in place to the value its
    own shooting-method solve converges on, which is not the archived row's
    recorded value to float precision: a real solve of the resumed row
    lands about 1% away from what was recorded for it originally. The
    schema check must therefore be run against the row's post-solve state,
    not its pre-solve archived value, and this is what any resume path
    that regenerates the mesh must do before Aragog reads it.
    """
    try:
        config = read_config_object(_CONFIG_PATH)
    except (UnknownConfigKeyError, ValueError) as exc:
        pytest.skip(f"archived config does not match this checkout's schema: {exc}")
    hf_all = ReadHelpfileFromCSV(_REFERENCE_RUN_DIR)
    hf_row = hf_all.iloc[_RESUMED_ROW_INDEX].to_dict()
    assert hf_row['Time'] == pytest.approx(_RESUMED_ROW_TIME_YR)
    r_int_before_solve = hf_row['R_int']

    os.makedirs(tmp_path / 'data')
    zalmoxis_solver(config, str(tmp_path), hf_row)

    regenerated_path = tmp_path / 'data' / 'zalmoxis_output.dat'
    data = np.loadtxt(regenerated_path)

    # Consistent with the row's post-solve state, checked against the file's
    # own recomputed mass integral rather than the solver's in-memory one.
    result = validate_zalmoxis_output_schema(str(regenerated_path), hf_row)
    assert result is None
    assert data[-1, 0] == pytest.approx(hf_row['R_int'], rel=1e-6)

    # The solve is real physics, not a passthrough of the archived value.
    solve_drift = abs(hf_row['R_int'] - r_int_before_solve) / r_int_before_solve
    assert solve_drift > 1e-4

    # The same row's schema check rejects the run's own stale, later-state
    # mesh -- the file left on disk before regeneration -- demonstrating
    # the check discriminates a wrong mesh from a right one rather than
    # merely confirming Zalmoxis produced some file.
    with pytest.raises(RuntimeError, match='top-of-mantle'):
        validate_zalmoxis_output_schema(_STALE_MESH_PATH, hf_row)
