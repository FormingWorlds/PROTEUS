"""Lockstep tests for the CHILI intercomparison generator.

The tutorial configs are the base of the CHILI cases
(tools/chili_generate.py). These tests pin that contract:

- each generated case equals its tutorial base on every config field
  outside its declared delta list, via the real config loader;
- the grid-spec inventory axes match the standalone grid configs under
  input/tutorials/chili_grid/;
- every CHILI config carries the default stop block plus PROTOCOL_STOP,
  and PROTOCOL_STOP holds the protocol values;
- the committed files under input/chili/intercomp/ equal a fresh
  regeneration, so a tutorial retune without regeneration fails CI.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import importlib.util
import logging
import re
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
    appears here as an undeclared difference. The nominal cases also pin
    the boundary from the other side: they differ from their tutorial in
    the output path, the redox state and the stop rule only, and the redox
    state differs in every case.
    """
    gen = _load_generator()
    for name, spec in gen.CASE_DELTAS.items():
        tut = _read_config(str(REPO / 'input' / 'tutorials' / spec['base']))
        case = _read_config(str(REPO / 'input' / 'chili' / 'intercomp' / f'{name}.toml'))
        diffs = set(_diff_paths(tut, case))
        assert 'outgas.fO2_shift_IW' in diffs, name
        declared = gen.case_deltas(name)
        undeclared = diffs - set(declared)
        assert not undeclared, f'{name}: differs from tutorial outside its deltas: {undeclared}'
        # Every declared delta must land in the loaded config with its
        # declared value: a generator regression that dropped a delta
        # would leave the tutorial value in place. Deltas that coincide
        # with the base value (a documented pin, e.g. tr1b age_ini) are
        # covered by the same direct comparison.
        for dotted, want in declared.items():
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
    # Discrimination: the nominal cases carry these deltas and no other.
    nominal = {'params.out.path', 'outgas.fO2_shift_IW', *gen.PROTOCOL_STOP}
    assert set(gen.case_deltas('earth')) == nominal
    assert set(gen.case_deltas('venus')) == nominal
    assert set(gen.CASE_DELTAS) == {'earth', 'venus', 'tr1b', 'tr1e', 'tr1a'}


def test_a_case_cannot_set_its_own_redox_state():
    """The redox state of the protocol is one value for all cases."""
    gen = _load_generator()
    assert 'outgas.fO2_shift_IW' not in gen.CASE_DELTAS['tr1a']['set']
    assert gen.case_deltas('tr1a')['outgas.fO2_shift_IW'] == pytest.approx(4.0)
    gen.CASE_DELTAS['tr1a']['set']['outgas.fO2_shift_IW'] = 3.0
    with pytest.raises(ValueError, match='tr1a: PROTOCOL_FO2_SHIFT_IW'):
        gen.generate_case('tr1a')


@pytest.mark.parametrize('key', _load_generator().PROTOCOL_STOP)
def test_a_case_cannot_set_its_own_stop_rule(key):
    """The stop rule of the protocol is one rule for all cases."""
    gen = _load_generator()
    assert key not in gen.CASE_DELTAS['venus']['set']
    assert gen.case_deltas('venus')[key] == gen.PROTOCOL_STOP[key]
    gen.CASE_DELTAS['venus']['set'][key] = 1
    with pytest.raises(ValueError, match=rf"venus: .*\['{re.escape(key)}'\]"):
        gen.generate_case('venus')


def test_every_chili_config_stops_only_by_the_protocol_rule():
    """The 5 cases and the 9 grid configs carry the default stop block plus PROTOCOL_STOP.

    The whole ``params.stop`` is compared, so a criterion turned on or off
    outside the protocol keys (solid.enabled, strict, stall) fails as well.
    """
    from proteus.config._params import StopParams

    gen = _load_generator()
    assert gen.PROTOCOL_STOP == {
        'params.stop.solid.phi_crit': pytest.approx(0.05, rel=1e-12),
        'params.stop.time.maximum': pytest.approx(1.0e9, rel=1e-12),
        'params.stop.radeqm.enabled': False,
        'params.stop.escape.enabled': False,
        'params.stop.iters.maximum': 1000000,
        'params.stop.clock.enabled': False,
    }
    want = attrs.asdict(StopParams())
    assert want['stall'] == {'enabled': True, 'maximum': 150}
    for dotted, value in gen.PROTOCOL_STOP.items():
        table, key = dotted.split('.')[2:]
        want[table][key] = value
    files = [REPO / 'input' / 'chili' / 'intercomp' / f'{n}.toml' for n in gen.CASE_DELTAS]
    files += sorted((REPO / 'input' / 'tutorials' / 'chili_grid').glob('earth_H*_C*.toml'))
    assert len(files) == 14
    for path in files:
        assert attrs.asdict(_read_config(str(path)).params.stop) == want, path.name


def test_protocol_values_are_fresh_items_in_each_case(tmp_path, monkeypatch):
    """A comment on a replaced tutorial line does not travel into the next case."""
    for name in ('tutorial_earth.toml', 'tutorial_venus.toml'):
        (tmp_path / name).write_text((REPO / 'input' / 'tutorials' / name).read_text())
    venus = tmp_path / 'tutorial_venus.toml'
    venus.write_text(venus.read_text().replace('maximum = 3e6', 'maximum = 3e6  # venus note'))
    gen = _load_generator()
    monkeypatch.setattr(gen, 'TUTORIALS', str(tmp_path))
    assert '# venus note' in venus.read_text()
    gen.generate_case('venus')
    assert '# venus note' not in gen.generate_case('tr1b')


@pytest.mark.parametrize('failure', ['case', 'grid'])
def test_a_refused_case_leaves_every_generated_file_as_it_was(tmp_path, failure):
    """A case or a grid spec that cannot be rendered stops ``main`` before any write."""
    gen = _load_generator()
    gen.OUTDIR = str(tmp_path)
    gen.main()
    files = sorted(tmp_path.iterdir())
    assert len(files) == 7
    for p in files:
        p.write_text('stale')
    if failure == 'case':
        gen.CASE_DELTAS['tr1a']['set']['outgas.fO2_shift_IW'] = 3.0
    else:
        gen.generate_grid = lambda planet: (_ for _ in ()).throw(ValueError('venus grid'))
    with pytest.raises(ValueError, match='tr1a' if failure == 'case' else 'venus grid'):
        gen.main()
    assert [p.read_text() for p in sorted(tmp_path.iterdir())] == ['stale'] * 7


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
        assert cfg['outgas']['fO2_shift_IW'] == pytest.approx(gen.PROTOCOL_FO2_SHIFT_IW), f.name
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
