from __future__ import annotations

import pytest

from tests.test_proteus import (
    _make_resume_checkpoint_df,
    _make_resume_main_loop_proteus,
    _run_resumed_loop_until_stop,
    _StopAfterAtmosphereCall,
)

pytestmark = [pytest.mark.unit]


def test_probe(tmp_path):
    p = _make_resume_main_loop_proteus(tmp_path, interior_module='spider', miscibility=True)
    hf_df = _make_resume_checkpoint_df()
    captured = []

    def fi(*a, **k):
        a[3]['T_magma'] = 3456.0

    def fa(*a, **k):
        r = a[8]
        captured.append(
            {
                k: r[k]
                for k in (
                    'T_magma',
                    'T_surf',
                    'P_surf',
                    'R_int',
                    'R_solvus',
                    'T_solvus',
                    'P_solvus',
                )
            }
        )
        raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, fi, fa)
    print('CAPTURED', captured)
