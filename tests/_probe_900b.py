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
    p.config.atmos_clim.surf_state = 'skin'
    hf_df = _make_resume_checkpoint_df()
    hf_df['P_surf'] = 250.0
    n = {'a': 0}
    seen = []

    def fi(*a, **k):
        r = a[3]
        seen.append(('interior_in_T_surf', r['T_surf']))
        r['T_magma'] = 3456.0 - 10 * n['a']
        r['R_solvus'] = 0.9 * r['R_int']
        r['T_solvus'] = 3700.0
        r['P_solvus'] = 2e10

    def fa(*a, **k):
        r = a[8]
        seen.append(('atm_in_T_surf', r['T_surf']))
        r['T_surf'] = 2000.0 + n['a']  # the atmosphere's solved skin temperature
        r['F_atm'] = 1e5
        n['a'] += 1
        if n['a'] == 4:
            raise _StopAfterAtmosphereCall

    _run_resumed_loop_until_stop(p, hf_df, fi, fa)
    print('SEEN', seen)
    print(
        'COMMITTED T_surf',
        list(p.hf_all['T_surf'].iloc[-4:]),
        'F_atm',
        list(p.hf_all['F_atm'].iloc[-4:]),
    )
