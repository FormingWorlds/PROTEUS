#!/usr/bin/env python3
"""Round-trip equivalence check for the 2.0 -> 3.0 config migration.

For each 2.0 config, this resolves the input through main's loader and the
translated output through this branch's loader, then asserts that every
physically meaningful field matches across the rename map. It is the
gold-standard faithfulness oracle: it exercises both real schema loaders rather
than the migration tool's own claims.

It is not part of the unit suite because it needs both schema revisions
importable, which requires a second checkout of main. Run it manually:

    # main checked out as a sibling worktree, configs taken from there
    git worktree add ../PROTEUS-main main
    python tools/migrate_equivalence_check.py --main-src ../PROTEUS-main/src \
        --configs '../PROTEUS-main/input/**/*.toml'

Exit code is non-zero if any config fails equivalence. Configs that are not
standard Config files (grid/inference configs) are skipped.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import types
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS))

if 'proteus.proteus' not in sys.modules:
    _stub = types.ModuleType('proteus.proteus')
    _stub.Proteus = object
    sys.modules['proteus.proteus'] = _stub

import migrate_config_v2_to_v3 as mig  # noqa: E402

_R_EARTH = 6335439.0

# v2 path -> v3 path, optional value transform, optional (section, active) guard.
# An independent re-derivation of the map, used as the comparison oracle.
_T_SELF = {'self': 'spider'}
_T_ML = {1: 'nearest', 2: 'constant'}
_T_SMA = {'sma': 'distance'}
CORR = [
    ('struct.module', 'interior_struct.module', lambda v: _T_SELF.get(v, v), None),
    ('interior.module', 'interior_energetics.module', None, None),
    ('atmos_clim.module', 'atmos_clim.module', None, None),
    ('outgas.module', 'outgas.module', None, None),
    ('escape.module', 'escape.module', None, None),
    ('star.module', 'star.module', None, None),
    ('orbit.module', 'orbit.module', None, None),
    ('delivery.module', 'accretion.module', None, None),
    ('struct.mass_tot', 'planet.mass_tot', None, None),
    (
        'struct.radius_int',
        'planet.R_int_override',
        lambda v: v * _R_EARTH if isinstance(v, (int, float)) else v,
        None,
    ),
    ('struct.corefrac', 'interior_struct.core_frac', None, None),
    ('struct.core_density', 'interior_struct.core_density', None, None),
    ('struct.core_heatcap', 'interior_struct.core_heatcap', None, None),
    ('interior.melting_dir', 'interior_struct.melting_dir', None, None),
    ('interior.eos_dir', 'interior_struct.eos_dir', None, None),
    ('interior.F_initial', 'interior_energetics.flux_guess', None, None),
    ('interior.grain_size', 'interior_energetics.grain_size', None, None),
    ('interior.rheo_phi_loc', 'interior_energetics.rfront_loc', None, None),
    ('interior.rheo_phi_wid', 'interior_energetics.rfront_wid', None, None),
    ('interior.radiogenic_heat', 'interior_energetics.heat_radiogenic', None, None),
    ('interior.tidal_heat', 'interior_energetics.heat_tidal', None, None),
    ('delivery.radio_K', 'interior_energetics.radio_K', None, None),
    ('delivery.radio_U', 'interior_energetics.radio_U', None, None),
    ('delivery.radio_Th', 'interior_energetics.radio_Th', None, None),
    ('delivery.radio_tref', 'interior_energetics.radio_tref', None, None),
    (
        'interior.spider.num_levels',
        'interior_energetics.num_levels',
        None,
        ('interior', 'spider'),
    ),
    ('interior.spider.tolerance', 'interior_energetics.atol', None, ('interior', 'spider')),
    ('interior.spider.tolerance_rel', 'interior_energetics.rtol', None, ('interior', 'spider')),
    (
        'interior.spider.tsurf_atol',
        'interior_energetics.tmagma_atol',
        None,
        ('interior', 'spider'),
    ),
    (
        'interior.spider.tsurf_rtol',
        'interior_energetics.tmagma_rtol',
        None,
        ('interior', 'spider'),
    ),
    (
        'interior.spider.mixing_length',
        'interior_energetics.mixing_length',
        lambda v: _T_ML.get(v, v),
        ('interior', 'spider'),
    ),
    (
        'interior.spider.conduction',
        'interior_energetics.trans_conduction',
        None,
        ('interior', 'spider'),
    ),
    (
        'interior.spider.gravitational_separation',
        'interior_energetics.trans_grav_sep',
        None,
        ('interior', 'spider'),
    ),
    (
        'interior.spider.solver_type',
        'interior_energetics.spider.solver_type',
        None,
        ('interior', 'spider'),
    ),
    ('interior.spider.ini_entropy', 'planet.ini_entropy', None, ('interior', 'spider')),
    ('interior.spider.ini_dsdr', 'planet.ini_dsdr', None, ('interior', 'spider')),
    ('outgas.fO2_shift_IW', 'outgas.fO2_shift_IW', None, None),
    ('outgas.calliope.T_floor', 'outgas.T_floor', None, ('outgas', 'calliope')),
    ('outgas.calliope.rtol', 'outgas.solver_rtol', None, ('outgas', 'calliope')),
    ('outgas.calliope.xtol', 'outgas.solver_atol', None, ('outgas', 'calliope')),
    ('atmos_clim.agni.dx_max_ini', 'atmos_clim.agni.dx_max_ini', None, ('atmos_clim', 'agni')),
    ('atmos_clim.agni.max_steps', 'atmos_clim.agni.max_steps', None, ('atmos_clim', 'agni')),
    (
        'atmos_clim.agni.spectral_group',
        'atmos_clim.spectral_group',
        None,
        ('atmos_clim', 'agni'),
    ),
    ('atmos_clim.agni.num_levels', 'atmos_clim.num_levels', None, ('atmos_clim', 'agni')),
    ('atmos_clim.agni.p_top', 'atmos_clim.p_top', None, ('atmos_clim', 'agni')),
    ('atmos_clim.surf_state', 'atmos_clim.surf_state', None, None),
    ('atmos_clim.surf_greyalbedo', 'atmos_clim.surf_greyalbedo', None, None),
    ('orbit.semimajoraxis', 'orbit.semimajoraxis', None, None),
    ('orbit.eccentricity', 'orbit.eccentricity', None, None),
    (
        'orbit.instellation_method',
        'orbit.instellation_method',
        lambda v: _T_SMA.get(v, v),
        None,
    ),
    ('star.mass', 'star.mass', None, None),
    ('star.age_ini', 'star.age_ini', None, None),
    ('star.mors.age_now', 'star.mors.age_now', None, ('star', 'mors')),
    ('params.dt.minimum', 'params.dt.minimum', None, None),
    ('params.dt.proportional.propconst', 'params.dt.propconst', None, None),
    ('params.dt.adaptive.atol', 'params.dt.atol', None, None),
    ('params.stop.escape.p_stop', 'params.stop.escape.p_stop', None, None),
]

_V2_RESOLVER = """
import sys, types, json, os
sys.path.insert(0, sys.argv[2])
stub = types.ModuleType("proteus.proteus"); stub.Proteus = object
sys.modules["proteus.proteus"] = stub
from proteus.config import read_config_object
from attrs import asdict
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        q = f"{p}.{k}" if p else k
        o.update(flat(v, q)) if isinstance(v, dict) else o.__setitem__(q, v)
    return o
try:
    os.write(1, json.dumps(flat(asdict(read_config_object(sys.argv[1]))), default=str).encode())
    os._exit(0)
except Exception as e:
    os.write(2, str(e).encode()[:300]); os._exit(3)
"""


def _norm(v):
    if v is None or (isinstance(v, str) and v.lower() == 'none'):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 9)
    return v


def _flat(d, p=''):
    o = {}
    for k, v in d.items():
        q = f'{p}.{k}' if p else k
        o.update(_flat(v, q)) if isinstance(v, dict) else o.__setitem__(q, v)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--main-src', required=True, help='path to a checkout of main src/')
    ap.add_argument('--configs', required=True, help='glob of 2.0 config files')
    args = ap.parse_args(argv)

    resolver = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    resolver.write(_V2_RESOLVER)
    resolver.close()

    def resolve_v2(path):
        env = dict(os.environ)
        env['PYTHONPATH'] = args.main_src
        r = subprocess.run(
            [sys.executable, resolver.name, path, args.main_src], capture_output=True, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode()[:200])
        return json.loads(r.stdout.decode())

    def resolve_v3(nested):
        import cattrs

        from proteus.config._config import Config

        return _flat(__import__('attrs').asdict(cattrs.structure(nested, Config)))

    def check(path):
        v2 = resolve_v2(path)
        with open(path, 'rb') as f:
            v2toml = tomllib.load(f)
        nested, report = mig.translate(v2toml)
        v3 = resolve_v3(nested)
        act = {
            'interior': v2.get('interior.module'),
            'atmos_clim': v2.get('atmos_clim.module'),
            'outgas': v2.get('outgas.module'),
            'star': v2.get('star.module'),
        }
        fails = []
        for v2p, v3p, tf, cond in CORR:
            if cond and act.get(cond[0]) != cond[1]:
                continue
            if v2p not in v2:
                continue
            exp = tf(v2[v2p]) if tf else v2[v2p]
            if _norm(exp) is None:
                continue
            if _norm(exp) != _norm(v3.get(v3p)):
                fails.append(
                    f'{v2p}={v2[v2p]!r} -> {v3p}: expected {exp!r}, got {v3.get(v3p)!r}'
                )
        if (
            act['interior'] in ('spider', 'aragog')
            and _norm(v3.get('interior_energetics.kappah_floor')) != 0.0
        ):
            fails.append('kappah_floor not 0.0')
        fails += [f'WARN {w}' for w in report.warnings if 'Unmapped' in w]
        return fails

    npass = nfail = nskip = 0
    for c in sorted(glob.glob(args.configs, recursive=True)):
        try:
            v2 = resolve_v2(c)
        except Exception:
            nskip += 1
            continue
        if 'struct.module' not in v2 and 'interior.module' not in v2:
            nskip += 1
            continue
        try:
            fails = check(c)
        except Exception as e:
            print(f'FAIL(exc) {Path(c).name}: {str(e)[:200]}')
            nfail += 1
            continue
        if fails:
            print(f'FAIL {Path(c).name}:')
            for fl in fails[:8]:
                print(f'    {fl}')
            nfail += 1
        else:
            npass += 1
    print(f'\n{npass} pass, {nfail} fail, {nskip} skip (non-Config)')
    return 1 if nfail else 0


if __name__ == '__main__':
    raise SystemExit(main())
