"""Lockstep tests for the CHILI intercomparison generator.

The tutorial configs are the single source of truth for the CHILI
cases (tools/chili_generate.py). These tests pin that contract:

- each generated case equals its tutorial base on every config field
  outside its declared delta list, via the real config loader;
- the grid-spec inventory axes match the standalone grid configs under
  input/tutorials/chili_grid/;
- the committed files under input/chili/intercomp/ equal a fresh
  regeneration, so a tutorial retune without regeneration fails CI.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import attrs
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO = Path(__file__).resolve().parents[2]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        'chili_generate', REPO / 'tools' / 'chili_generate.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_config(path):
    logging.disable(logging.WARNING)
    try:
        from proteus.config import read_config_object

        return read_config_object(path)
    finally:
        logging.disable(logging.NOTSET)


def _diff_paths(a, b, path=''):
    """Dotted paths of every leaf field that differs between two configs."""
    out = []
    if attrs.has(type(a)) and attrs.has(type(b)) and type(a) is type(b):
        for f in attrs.fields(type(a)):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            out += _diff_paths(va, vb, f'{path}.{f.name}' if path else f.name)
    elif a != b:
        out.append(path)
    return out


def test_generated_cases_match_tutorials_outside_their_deltas():
    """Every case equals its tutorial base except the declared deltas.

    The contract that keeps the intercomparison and the tutorials in
    lockstep: a physics field changed in only one of the two surfaces
    appears here as an undeclared difference. The earth case also pins
    the boundary from the other side: its only difference is the
    output path, so the nominal CHILI Earth IS the tutorial Earth.
    """
    gen = _load_generator()
    for name, spec in gen.CASE_DELTAS.items():
        tut = _read_config(str(REPO / 'input' / 'tutorials' / spec['base']))
        case = _read_config(str(REPO / 'input' / 'chili' / 'intercomp' / f'{name}.toml'))
        diffs = set(_diff_paths(tut, case))
        declared = set(spec['set'].keys())
        undeclared = diffs - declared
        assert not undeclared, f'{name}: differs from tutorial outside its deltas: {undeclared}'
        # Every declared delta must land in the loaded config with its
        # declared value: a generator regression that dropped a delta
        # would leave the tutorial value in place. Deltas that coincide
        # with the base value (a documented pin, e.g. tr1b age_ini) are
        # covered by the same direct comparison.
        for dotted, want in spec['set'].items():
            got = case
            for part in dotted.split('.'):
                got = getattr(got, part)
            if want == 'none':
                # The config loader maps the TOML 'none' sentinel to None.
                assert got is None, f'{name}: {dotted}={got!r}, want the none sentinel'
            elif isinstance(want, float):
                assert got == pytest.approx(want, rel=1e-12), f'{name}: {dotted}={got!r}'
            else:
                assert got == want, f'{name}: {dotted}={got!r}, want {want!r}'
    # Discrimination: the nominal cases are single-delta by design.
    assert set(gen.CASE_DELTAS['earth']['set']) == {'params.out.path'}
    assert set(gen.CASE_DELTAS['venus']['set']) == {'params.out.path'}


def test_grid_axes_match_standalone_grid_configs():
    """The grid-spec inventory axes equal the standalone grid configs.

    The 3x3 standalone configs under input/tutorials/chili_grid/ carry
    exactly the protocol H and C budgets that the grid specs sweep; a
    change to either surface without the other breaks the pairing.
    """
    import tomllib

    gen = _load_generator()
    h_axis = sorted(gen.GRID_AXES['planet.elements.H_budget'])
    c_axis = sorted(gen.GRID_AXES['planet.elements.C_budget'])

    grid_dir = REPO / 'input' / 'tutorials' / 'chili_grid'
    files = sorted(grid_dir.glob('earth_H*_C*.toml'))
    # The full cross product must be present: 3 x 3 cases.
    assert len(files) == len(h_axis) * len(c_axis)

    budgets = set()
    for f in files:
        with open(f, 'rb') as hdl:
            cfg = tomllib.load(hdl)
        ele = cfg['planet']['elements']
        budgets.add((float(ele['H_budget']), float(ele['C_budget'])))
    expected = {(h, c) for h in h_axis for c in c_axis}
    assert budgets == expected, (
        f'grid configs do not span the protocol axes: {budgets ^ expected}'
    )


def test_committed_intercomp_files_match_regeneration():
    """The committed intercomparison files equal a fresh regeneration.

    Retuning a tutorial without rerunning tools/chili_generate.py
    leaves stale committed configs; this comparison fails CI until the
    regeneration is committed. The comparison is on the parsed TOML
    content, so a serializer formatting change alone cannot fail it.
    """
    import tomllib

    gen = _load_generator()
    outdir = REPO / 'input' / 'chili' / 'intercomp'
    stale = []
    for name in gen.CASE_DELTAS:
        committed = tomllib.loads((outdir / f'{name}.toml').read_text())
        if committed != tomllib.loads(gen.generate_case(name)):
            stale.append(f'{name}.toml')
    for planet in ('earth', 'venus'):
        committed = tomllib.loads((outdir / f'{planet}.grid.toml').read_text())
        if committed != tomllib.loads(gen.generate_grid(planet)):
            stale.append(f'{planet}.grid.toml')
    assert not stale, f'stale committed configs, rerun tools/chili_generate.py: {stale}'
    # Discrimination: the comparison must be able to fail on a physics
    # field outside the delta list, not just on the output path.
    perturbed = gen.generate_case('earth').replace(
        'semimajoraxis = 1.0', 'semimajoraxis = 1.05'
    )
    assert perturbed != gen.generate_case('earth')  # the perturbation landed
    assert tomllib.loads(perturbed) != tomllib.loads((outdir / 'earth.toml').read_text())
